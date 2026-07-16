#!/usr/bin/env python3
"""Query an Ollama-served LLM for FIFA World Cup 2026 score predictions.

Protocol (per model, per match):
  1 greedy generation  (temperature 0,   seed = --greedy-seed)
  N sampled generations (temperature --temperature, seeds = seed_base+1 .. seed_base+N)

Every raw API response is appended (JSONL, flushed immediately) to
  <rawdir>/<model>/<match_id>.jsonl
The aggregated prediction per match is appended to
  <outdir>/<model>.jsonl
so re-runs never destroy history; score.py uses the latest record per match.

Stdlib only - no pip installs needed on the cluster.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are an expert football (soccer) forecaster. You know the teams' squads, "
    "playing styles, form, and historical head-to-head records. Respond with a single "
    "JSON object matching the requested schema and nothing else."
)

USER_TEMPLATE = (
    "FIFA World Cup 2026 - {stage} ({match_id}).\n"
    "Match: {home} vs {away}\n"
    "Venue: {venue}\n"
    "Kickoff (UTC): {kickoff_utc}\n\n"
    "Predict the full-time score after 90 minutes of regulation only (exclude extra time "
    "and penalties). Regardless of your scoreline, also state which team advances from this "
    "tie (if you predict a draw, that means the team you expect to win in extra time or on "
    "penalties; for a third-place match or final, the team that lifts the result). "
    "Keep one_line_reasoning under 40 words."
)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sanitize(tag):
    return tag.replace("/", "_").replace(":", "_")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def http_json(host, path, payload=None, timeout=1800):
    """POST (or GET if payload is None) JSON to the Ollama server; return parsed JSON."""
    url = host.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_schema(match):
    """JSON schema for Ollama structured outputs; 'advances' is an enum of the two teams."""
    return {
        "type": "object",
        "properties": {
            "home_goals": {"type": "integer", "minimum": 0, "maximum": 15},
            "away_goals": {"type": "integer", "minimum": 0, "maximum": 15},
            "advances": {"type": "string", "enum": [match["home"], match["away"]]},
            "one_line_reasoning": {"type": "string"},
        },
        "required": ["home_goals", "away_goals", "advances"],
    }


def extract_json(text):
    """Fallback parser: first balanced {...} block in free text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def validate_parsed(parsed, match):
    if not isinstance(parsed, dict):
        return None
    try:
        h = int(parsed["home_goals"])
        a = int(parsed["away_goals"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= h <= 15 and 0 <= a <= 15):
        return None
    adv = parsed.get("advances")
    if adv not in (match["home"], match["away"]):
        # tolerate loose matches like "France " or "the France team"
        adv_l = str(adv).lower() if adv is not None else ""
        if match["home"].lower() in adv_l:
            adv = match["home"]
        elif match["away"].lower() in adv_l:
            adv = match["away"]
        else:
            adv = None
    return {
        "home_goals": h,
        "away_goals": a,
        "advances": adv,
        "one_line_reasoning": str(parsed.get("one_line_reasoning", ""))[:500],
    }


def one_generation(host, model, match, seed, temperature, num_ctx, raw_fh, kind, retries=3):
    """Run one chat completion; log the raw exchange; return validated dict or None."""
    user_prompt = USER_TEMPLATE.format(**match)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": build_schema(match),
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": 4096,
        },
    }
    last_err = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        record = {
            "ts": now_iso(),
            "kind": kind,
            "model": model,
            "match_id": match["match_id"],
            "seed": seed,
            "temperature": temperature,
            "attempt": attempt,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": sha256(SYSTEM_PROMPT + "\n" + user_prompt),
        }
        try:
            resp = http_json(host, "/api/chat", payload)
            record["wall_seconds"] = round(time.time() - t0, 2)
            record["response"] = resp  # full raw response incl. token counts/durations
            content = resp.get("message", {}).get("content", "")
            parsed = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = extract_json(content)
            valid = validate_parsed(parsed, match)
            record["parsed"] = valid
            record["parse_ok"] = valid is not None
            raw_fh.write(json.dumps(record) + "\n")
            raw_fh.flush()
            if valid is not None:
                return valid, record
            last_err = "unparseable response"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            record["wall_seconds"] = round(time.time() - t0, 2)
            record["error"] = repr(e)
            raw_fh.write(json.dumps(record) + "\n")
            raw_fh.flush()
            last_err = repr(e)
        print(f"    attempt {attempt}/{retries} failed ({last_err}); retrying...",
              flush=True)
        time.sleep(min(10 * attempt, 30))
    print(f"    GIVING UP on {match['match_id']} seed={seed}: {last_err}", flush=True)
    return None, None


def aggregate(generations, greedy, match):
    """Modal scoreline across all generations.

    Tie-breaks, in order: higher count, equals the greedy prediction,
    lower total goals, lower away goals. 'advances' is the majority vote;
    if the aggregated scoreline is not a draw it is forced to the winner
    (flagged as advances_overridden).
    """
    scores = Counter((g["home_goals"], g["away_goals"]) for g in generations)
    greedy_score = (greedy["home_goals"], greedy["away_goals"]) if greedy else None

    def rank(item):
        (h, a), count = item
        return (-count, 0 if (h, a) == greedy_score else 1, h + a, a)

    (h, a), count = sorted(scores.items(), key=rank)[0]

    adv_votes = Counter(g["advances"] for g in generations if g["advances"])
    adv = adv_votes.most_common(1)[0][0] if adv_votes else None
    overridden = False
    if h > a and adv != match["home"]:
        adv, overridden = match["home"], adv is not None
    elif a > h and adv != match["away"]:
        adv, overridden = match["away"], adv is not None

    n = len(generations)
    outcome_probs = {
        "home_win": sum(1 for g in generations if g["home_goals"] > g["away_goals"]) / n,
        "draw": sum(1 for g in generations if g["home_goals"] == g["away_goals"]) / n,
        "away_win": sum(1 for g in generations if g["home_goals"] < g["away_goals"]) / n,
    }
    return {
        "home_goals": h,
        "away_goals": a,
        "modal_count": count,
        "n_generations": n,
        "advances": adv,
        "advances_overridden": overridden,
        "outcome_probs": outcome_probs,
        "score_distribution": {f"{k[0]}-{k[1]}": v for k, v in scores.most_common()},
        "advance_votes": dict(adv_votes),
        "mean_home_goals": round(sum(g["home_goals"] for g in generations) / n, 3),
        "mean_away_goals": round(sum(g["away_goals"] for g in generations) / n, 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Ollama model tag, e.g. gpt-oss:20b")
    p.add_argument("--matches", default="data/matches.json")
    p.add_argument("--stage", default="all",
                   help="match_ids (QF1,QF2,...) or prefixes (QF, SF, TPP, F) or 'all'; "
                        "separate with ',' or '+' ('+' survives sbatch --export, which eats commas)")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--greedy-seed", type=int, default=42)
    p.add_argument("--sample-seed-base", type=int, default=1000)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--outdir", default="results/predictions")
    p.add_argument("--rawdir", default="logs/raw")
    args = p.parse_args()

    host = args.host if args.host.startswith("http") else "http://" + args.host

    version = http_json(host, "/api/version", timeout=30)
    try:
        show = http_json(host, "/api/show", {"model": args.model}, timeout=120)
        model_info = {
            "details": show.get("details"),
            "parameters": show.get("parameters"),
            "digest": show.get("digest") or show.get("modelfile", "")[:0] or None,
        }
    except Exception as e:
        model_info = {"error": f"api/show failed: {e!r}"}

    with open(args.matches) as f:
        matches_doc = json.load(f)

    wanted = [w.strip() for w in args.stage.replace("+", ",").split(",")]
    def selected(m):
        if args.stage == "all":
            return True
        return any(m["match_id"] == w or m["match_id"].startswith(w) for w in wanted)

    matches = [m for m in matches_doc["matches"] if selected(m)]
    runnable = [m for m in matches if m["home"] != "TBD" and m["away"] != "TBD"]
    skipped = [m["match_id"] for m in matches if m not in runnable]
    if skipped:
        print(f"Skipping TBD matches: {', '.join(skipped)}", flush=True)
    if not runnable:
        print("Nothing to run.", flush=True)
        sys.exit(0)

    mtag = sanitize(args.model)
    os.makedirs(args.outdir, exist_ok=True)
    rawdir = os.path.join(args.rawdir, mtag)
    os.makedirs(rawdir, exist_ok=True)
    out_path = os.path.join(args.outdir, mtag + ".jsonl")

    run_meta = {
        "run_ts": now_iso(),
        "model": args.model,
        "ollama_version": version.get("version"),
        "model_info": model_info,
        "git_commit": git_commit(),
        "prompt_version": PROMPT_VERSION,
        "samples": args.samples,
        "temperature": args.temperature,
        "greedy_seed": args.greedy_seed,
        "sample_seed_base": args.sample_seed_base,
        "num_ctx": args.num_ctx,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    print(json.dumps({"run_meta": run_meta}, indent=2), flush=True)

    failures = 0
    with open(out_path, "a") as out_fh:
        for match in runnable:
            print(f"\n=== {match['match_id']}: {match['home']} vs {match['away']} ===",
                  flush=True)
            raw_path = os.path.join(rawdir, match["match_id"] + ".jsonl")
            with open(raw_path, "a") as raw_fh:
                t0 = time.time()
                greedy, _ = one_generation(
                    host, args.model, match, args.greedy_seed, 0.0,
                    args.num_ctx, raw_fh, "greedy")
                gens = [greedy] if greedy else []
                for i in range(1, args.samples + 1):
                    seed = args.sample_seed_base + i
                    g, _ = one_generation(
                        host, args.model, match, seed, args.temperature,
                        args.num_ctx, raw_fh, "sample")
                    if g:
                        gens.append(g)
                        print(f"    seed {seed}: {g['home_goals']}-{g['away_goals']} "
                              f"(advances: {g['advances']})", flush=True)

            if not gens:
                print(f"  !! no valid generations for {match['match_id']}", flush=True)
                failures += 1
                continue

            agg = aggregate(gens, greedy, match)
            rec = {
                "ts": now_iso(),
                "model": args.model,
                "match_id": match["match_id"],
                "home": match["home"],
                "away": match["away"],
                "prediction": agg,
                "greedy_prediction": (
                    {"home_goals": greedy["home_goals"],
                     "away_goals": greedy["away_goals"],
                     "advances": greedy["advances"]} if greedy else None),
                "wall_seconds": round(time.time() - t0, 1),
                "run_meta": run_meta,
            }
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            print(f"  => aggregated: {agg['home_goals']}-{agg['away_goals']} "
                  f"({agg['modal_count']}/{agg['n_generations']} votes), "
                  f"advances: {agg['advances']}", flush=True)

    print(f"\nDone. Aggregated predictions appended to {out_path}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
