#!/usr/bin/env python3
"""Refresh the data-driven parts of runlog.html in place:
the SLURM jobs table (#jobs-body, from sacct), the leaderboard table
(#lb-body, #lb-title, #kpi-matches, #kpi-leader*, from results/scores/).
Narrative sections (timeline, lessons) are edited by hand.

Run from the repo root after re-scoring:  python3 scripts/update_runlog.py
"""
import html
import json
import os
import re
import subprocess

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

NOTES = {
    "45404": ("Setup", "Connectivity probe: compute nodes reach github.com & ollama.com — enabled the compute-node-only design"),
    "45408": ("Setup", "CPU-only prefetch of 9 models' weights (~110 GB) to shared /home; no GPU held"),
    "45428": ("QF", "FAILED as expected-risk: minimax-m2.7 is a cloud-only tag — 'pull model manifest: file does not exist'; moved to not_feasible"),
}
for j in ("47738", "47739", "47740", "48076", "48077"):
    NOTES[j] = ("Aux", "Auxiliary job from the dashboard-dev session; output not in this repo's logs/")

def phase_note(jid):
    if jid in NOTES:
        return NOTES[jid]
    n = int(jid)
    if 45417 <= n <= 45427: return ("QF", "QF predictions (parallel submission — deadline mode)")
    if 47727 <= n <= 47737: return ("SF", "SF predictions (chained queue)")
    if 48065 <= n <= 48075: return ("TPP/F try", "Nothing to run: TPP/Final teams were still TBD in matches.json — predict.py skipped by design")
    if 48110 <= n <= 48120: return ("TPP", "TPP predictions (chained queue; STAGE lost ',F' to the sbatch comma bug)")
    if 48123 <= n <= 48133: return ("F", "Final predictions (chained behind the TPP queue via --after)")
    return ("", "")

sacct = subprocess.check_output([
    "sacct", "-u", os.environ.get("USER", "ehassan"), "-S", "2026-07-08", "-X", "-n", "-P",
    "--format=JobID,JobName,State,Start,Elapsed,NodeList,AllocTRES"]).decode()
job_rows = []
for line in sacct.strip().split("\n"):
    jid, name, state, start, elapsed, node, tres = line.split("|")
    gpus = "0"
    for part in tres.split(","):
        if part.startswith("gres/gpu="):
            gpus = part.split("=")[1]
    phase, note = phase_note(jid)
    cls = {"COMPLETED": "ok", "FAILED": "bad", "RUNNING": "run", "PENDING": "pend"}.get(state.split()[0], "")
    job_rows.append(
        f'<tr><td>{jid}</td><td>{html.escape(name)}</td><td>{phase}</td>'
        f'<td>{node}</td><td>{gpus}</td><td>{start.replace("T", " ")}</td>'
        f'<td>{elapsed}</td><td><span class="st {cls}">{state}</span></td>'
        f'<td class="note">{note}</td></tr>')

doc = json.load(open("results/scores/leaderboard.json"))
lb = doc["leaderboard"]
n_scored = len(doc["meta"]["matches_scored"])
lb_rows = [
    f'<tr><td>{r["rank"]}</td><td class="mdl">{r["model"]}</td><td class="tot">{r["points"]}</td>'
    f'<td>{r["exact"]}</td><td>{r["goal_diff"]}</td><td>{r["tendency"]}</td><td>{r["miss"]}</td>'
    f'<td>{r["advance_bonus"]}</td><td>{r["mean_rps"]}</td><td>{r["mean_brier"]}</td>'
    f'<td>{r["mae_total_goals"]}</td></tr>' for r in lb]

page = open("runlog.html").read()

def replace_between(page, start_marker, end_marker, new, count_from=None):
    i = page.index(start_marker) + len(start_marker)
    j = page.index(end_marker, i)
    return page[:i] + "\n" + new + "\n" + page[j:]

page = replace_between(page, '<tbody id="jobs-body">', "</tbody>", "\n".join(job_rows))
page = replace_between(page, '<tbody id="lb-body">', "</tbody>", "\n".join(lb_rows))
page = re.sub(r'(<h2 id="lb-title">)[^<]*', rf'\g<1>Leaderboard after {n_scored} of 8 matches', page)
page = re.sub(r'(<div class="num" id="kpi-matches">)[^<]*', rf'\g<1>{n_scored} / 8', page)
page = re.sub(r'(<div class="num" id="kpi-leader">)[^<]*', rf'\g<1>{lb[0]["model"]}', page)
page = re.sub(r'(<div class="label" id="kpi-leader-label">)[^<]*',
              rf'\g<1>leader after {n_scored} matches · {lb[0]["points"]} pts', page)

open("runlog.html", "w").write(page)
print(f"runlog.html refreshed: {len(job_rows)} jobs, {len(lb_rows)} leaderboard rows, "
      f"{n_scored}/8 matches, leader {lb[0]['model']} ({lb[0]['points']} pts)")
