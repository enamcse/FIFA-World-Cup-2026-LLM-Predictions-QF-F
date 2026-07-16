# FIFA World Cup 2026 — LLM Predictions (QF → Final)

Benchmarking open-weight LLMs (served locally via [Ollama](https://ollama.com)) on
predicting the scorelines of the last 8 matches of the 2026 World Cup, run on the
[Stony Brook AI Institute cluster](https://ai.stonybrook.edu/resources/computingresources).

**Start here → [docs/RUNBOOK.md](docs/RUNBOOK.md)** for the step-by-step cluster instructions.

**Ground rule:** the login node is only used to edit files and run `sbatch`/`squeue`.
Everything else — installing Ollama, downloading weights, inference — happens inside
SLURM jobs on compute nodes (artifacts land on shared `/home`, so downloads happen once).

## Remaining fixtures

| ID  | Stage        | Match                   | Kickoff (UTC)      | Venue          |
|-----|--------------|-------------------------|--------------------|----------------|
| QF1 | Quarter-final| France vs Morocco       | Jul 9, 20:00       | Boston         |
| QF2 | Quarter-final| Spain vs Belgium        | Jul 10, 16:00      | Inglewood (LA) |
| QF3 | Quarter-final| Norway vs England       | Jul 11, 21:00      | Miami Gardens  |
| QF4 | Quarter-final| Argentina vs Switzerland| Jul 12, 00:00      | Kansas City    |
| SF1 | Semi-final   | QF1w vs QF2w            | Jul 14 (evening)   | Arlington      |
| SF2 | Semi-final   | QF3w vs QF4w            | Jul 15 (evening)   | Atlanta        |
| TPP | Third place  | SF1l vs SF2l            | Jul 18             | Miami Gardens  |
| F   | Final        | SF1w vs SF2w            | Jul 19 (evening)   | East Rutherford|

Predictions for a match are only credible if committed **before kickoff** — the git
commit timestamp is the tamper-proof receipt. QF1 kicks off **July 9, 20:00 UTC**.

## Prediction protocol

Per model, per match: **1 greedy** generation (temperature 0) + **10 sampled**
generations (temperature 0.7, fixed seeds 1001–1010). Models answer in structured
JSON (`home_goals`, `away_goals` after 90 minutes, plus `advances` — who goes
through if it's a draw). The final prediction is the **modal scoreline** across all
11 generations; the sample spread also gives each model an empirical probability
distribution over win/draw/loss, which we score probabilistically. Every raw API
response is logged verbatim to `logs/raw/`.

## Scoring system

Points per match on the aggregated prediction (90-minute score), Kicktipp-style:

| Result | Points |
|---|---|
| Exact score | 4 |
| Correct goal difference, wrong score (incl. wrong-score draws) | 3 |
| Correct winner only | 2 |
| Wrong | 0 |
| Bonus: predicted advancing team actually advances | +1 |

Max 5 points/match, 40 for the tournament. Tie-break and secondary metrics from the
11-generation distribution: **RPS** (ranked probability score) and **Brier** over
win/draw/loss, plus MAE on goals. Implemented in [src/score.py](src/score.py).

## Model queue (priority order)

Defined in [configs/models.json](configs/models.json); sized for the cluster's GPUs
(H100 80GB / RTX 8000 48GB / V100 32GB). Single-GPU models run first (best backfill
odds on a busy queue); multi-GPU models follow. Summary:

| # | Model | Why | Fits |
|---|-------|-----|------|
| 1 | gpt-oss:20b | fast reasoning model, pipeline shakedown | any GPU |
| 2 | gemma4:26b | newest Google open model | any GPU |
| 3 | deepseek-r1:32b | reasoning distill | any GPU |
| 4 | glm-4.7-flash | GLM family, local-runnable size | any GPU |
| 5 | mistral-small3.2 | Mistral representative | any GPU |
| 6 | qwen3.6:35b | strong recent open family | 48GB+ |
| 7 | phi4:14b | strong small model | any GPU |
| 8–9 | llama3.1:8b, llama3.2:3b | small/tiny baselines | any GPU |
| 10 | llama3.3:70b | Meta flagship at feasible size | 2× RTX 8000 or 1× H100 |
| 11 | gpt-oss:120b | within-family scaling comparison | 2× RTX 8000 or 1× H100 |

**GLM-5.2 and MiniMax-M2.7 exist on the Ollama library but only as `:cloud` tags** —
models hosted on Ollama's cloud with no downloadable weights, so they can't run on
cluster GPUs (and cloud inference is outside this benchmark's scope; minimax-m2.7's
bare tag was tried and fails — job 45428). DeepSeek-V4-Pro and Llama-405B don't fit
locally either — see `not_feasible` in the config for the arithmetic.

## Repo layout

```
configs/models.json       model queue: tags, priorities, per-model SLURM resources
data/matches.json         fixtures (edit TBD teams as rounds resolve)
data/actuals.json         fill in real results after each match
src/predict.py            Ollama client: greedy+sampled generations, full logging
src/score.py              scoring + leaderboard (md/csv/json)
slurm/predict.slurm       generic one-model GPU job (installs/pulls on the node)
slurm/prefetch.slurm      CPU-only job to pre-download weights on a compute node
slurm/run_queue.sh        submits the queue: chained by default, --parallel for deadlines
scripts/ensure_ollama.sh  pinned user-space Ollama install; called inside jobs only
tests/smoke_test.sh       end-to-end test with a mock server, no GPU needed
results/predictions/      aggregated predictions, one JSONL per model  (committed)
results/scores/           leaderboard outputs                          (committed)
logs/raw/                 every raw model response                     (committed)
logs/slurm/               SLURM job + ollama server logs               (committed)
```

Everything needed to reproduce a number — model digest, pinned Ollama version, seeds,
prompt hash, git commit, hostname, job ID — is embedded in every JSONL record.

## Dashboard & run log

**Live dashboard → <https://enamcse.github.io/FIFA-World-Cup-2026-LLM-Predictions-QF-F/>**

**Run log (full chronicle) → <https://enamcse.github.io/FIFA-World-Cup-2026-LLM-Predictions-QF-F/runlog.html>** —
every step of the project, day by day: the exact prompts, every SLURM job with its
ID/node/elapsed time/state, per-round predictions vs reality with the timing audit,
and the incidents-and-lessons list ([runlog.html](runlog.html)).

[index.html](index.html) is a self-contained static page that reads
`results/scores/leaderboard.json`, `data/matches.json`, `data/actuals.json`, and
`results/predictions/*.jsonl` at runtime — bracket, leaderboard, points-by-match
heatmap, and a per-match breakdown of every model's call (including locked-in
predictions for unplayed matches). It is deployed to GitHub Pages by
[.github/workflows/pages.yml](.github/workflows/pages.yml) on every push to
`main`, so committing new actuals + a re-scored leaderboard updates the site
automatically. To preview locally: `python3 -m http.server` from the repo root.
