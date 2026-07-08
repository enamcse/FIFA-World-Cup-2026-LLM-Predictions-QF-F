#!/usr/bin/env python3
"""Score model predictions against actual results and build a leaderboard.

Scoring per match (on the aggregated prediction, 90-minute score):
    4 pts  exact score
    3 pts  correct goal difference, wrong score (includes wrong-score draws)
    2 pts  correct tendency only (right winner, wrong margin)
    0 pts  otherwise
   +1 pt   the team predicted to advance actually advanced
   Max 5 pts per match.

Probabilistic metrics (from the empirical distribution over all generations):
    RPS   ranked probability score over (home win, draw, away win) - lower is better
    Brier multiclass Brier score - lower is better

Error metrics: MAE on home goals, away goals, and total goals.

Leaderboard order: total points desc, then mean RPS asc.

Reads the LATEST aggregated record per (model, match) from results/predictions/*.jsonl,
so re-running predictions simply supersedes older records without deleting history.
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone


def outcome(h, a):
    return "home_win" if h > a else ("away_win" if h < a else "draw")


def kicktipp_points(ph, pa, ah, aa):
    if (ph, pa) == (ah, aa):
        return 4, "exact"
    if ph - pa == ah - aa:
        return 3, "goal_diff"
    if outcome(ph, pa) == outcome(ah, aa):
        return 2, "tendency"
    return 0, "miss"


def rps(probs, actual):
    """Ranked probability score for ordered outcomes (home_win, draw, away_win)."""
    order = ["home_win", "draw", "away_win"]
    p = [probs.get(k, 0.0) for k in order]
    o = [1.0 if actual == k else 0.0 for k in order]
    cp = co = s = 0.0
    for i in range(2):  # cumulative over first n-1 categories
        cp += p[i]
        co += o[i]
        s += (cp - co) ** 2
    return s / 2


def brier(probs, actual):
    order = ["home_win", "draw", "away_win"]
    return sum((probs.get(k, 0.0) - (1.0 if actual == k else 0.0)) ** 2 for k in order)


def latest_records(pred_dir):
    """{model: {match_id: record}} keeping only the newest record per pair."""
    out = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                model = rec["model"]
                out.setdefault(model, {})
                prev = out[model].get(rec["match_id"])
                if prev is None or rec["ts"] >= prev["ts"]:
                    out[model][rec["match_id"]] = rec
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="results/predictions")
    p.add_argument("--actuals", default="data/actuals.json")
    p.add_argument("--out", default="results/scores")
    args = p.parse_args()

    with open(args.actuals) as f:
        actuals = json.load(f)
    scored_matches = sorted(
        mid for mid, a in actuals.items()
        if not mid.startswith("_") and a.get("home_goals_90") is not None
    )
    if not scored_matches:
        print("No matches have actual results yet - fill in data/actuals.json first.")
        return

    preds = latest_records(args.predictions)
    os.makedirs(args.out, exist_ok=True)

    rows, breakdown = [], {}
    for model, by_match in sorted(preds.items()):
        totals = {
            "points": 0, "advance_bonus": 0, "exact": 0, "goal_diff": 0,
            "tendency": 0, "miss": 0, "matches": 0, "missing": [],
        }
        maes_h, maes_a, maes_t, rpss, briers = [], [], [], [], []
        breakdown[model] = {}
        for mid in scored_matches:
            act = actuals[mid]
            ah, aa, adv = act["home_goals_90"], act["away_goals_90"], act.get("advanced")
            rec = by_match.get(mid)
            if rec is None:
                totals["missing"].append(mid)
                continue
            pred = rec["prediction"]
            ph, pa = pred["home_goals"], pred["away_goals"]
            pts, category = kicktipp_points(ph, pa, ah, aa)
            bonus = 1 if adv and pred.get("advances") == adv else 0
            totals["points"] += pts + bonus
            totals["advance_bonus"] += bonus
            totals[category] += 1
            totals["matches"] += 1
            maes_h.append(abs(ph - ah))
            maes_a.append(abs(pa - aa))
            maes_t.append(abs((ph + pa) - (ah + aa)))
            act_outcome = outcome(ah, aa)
            rpss.append(rps(pred["outcome_probs"], act_outcome))
            briers.append(brier(pred["outcome_probs"], act_outcome))
            breakdown[model][mid] = {
                "predicted": f"{ph}-{pa}",
                "actual": f"{ah}-{aa}",
                "category": category,
                "points": pts,
                "advance_bonus": bonus,
                "predicted_advances": pred.get("advances"),
                "actually_advanced": adv,
                "rps": round(rpss[-1], 4),
                "brier": round(briers[-1], 4),
                "outcome_probs": pred["outcome_probs"],
            }
        if totals["matches"] == 0:
            continue
        n = totals["matches"]
        rows.append({
            "model": model,
            "matches": n,
            "points": totals["points"],
            "avg_points": round(totals["points"] / n, 3),
            "exact": totals["exact"],
            "goal_diff": totals["goal_diff"],
            "tendency": totals["tendency"],
            "miss": totals["miss"],
            "advance_bonus": totals["advance_bonus"],
            "mae_home": round(sum(maes_h) / n, 3),
            "mae_away": round(sum(maes_a) / n, 3),
            "mae_total_goals": round(sum(maes_t) / n, 3),
            "mean_rps": round(sum(rpss) / n, 4),
            "mean_brier": round(sum(briers) / n, 4),
            "missing_matches": ",".join(totals["missing"]),
        })

    rows.sort(key=lambda r: (-r["points"], r["mean_rps"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    cols = ["rank", "model", "matches", "points", "avg_points", "exact", "goal_diff",
            "tendency", "miss", "advance_bonus", "mae_home", "mae_away",
            "mae_total_goals", "mean_rps", "mean_brier", "missing_matches"]

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches_scored": scored_matches,
    }
    with open(os.path.join(args.out, "leaderboard.json"), "w") as f:
        json.dump({"meta": meta, "leaderboard": rows, "breakdown": breakdown}, f, indent=2)
    with open(os.path.join(args.out, "leaderboard.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "leaderboard.md"), "w") as f:
        f.write(f"# Leaderboard ({', '.join(scored_matches)})\n\n")
        f.write(f"Generated {meta['generated_at']}\n\n")
        f.write("| " + " | ".join(cols[:-1]) + " |\n")
        f.write("|" + "---|" * (len(cols) - 1) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols[:-1]) + " |\n")

    print(f"Scored {len(scored_matches)} match(es) for {len(rows)} model(s).")
    print(f"Wrote leaderboard.{{json,csv,md}} to {args.out}/\n")
    widths = [max(len(str(r.get(c, ''))) for r in rows + [dict(zip(cols, cols))]) for c in cols[:-1]]
    print("  ".join(c.ljust(w) for c, w in zip(cols[:-1], widths)))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols[:-1], widths)))


if __name__ == "__main__":
    main()
