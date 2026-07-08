# FIFA World Cup 2026 — LLM Predictions (QF → Final)

Benchmarking open-weight LLMs (served locally via [Ollama](https://ollama.com)) on
predicting the scorelines of the last 8 matches of the 2026 World Cup, run on the
[Stony Brook AI Institute cluster](https://ai.stonybrook.edu/resources/computingresources).

**Start here → [docs/RUNBOOK.md](docs/RUNBOOK.md)** for the step-by-step cluster instructions.

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
(H100 80GB / RTX 8000 48GB / V100 32GB). Summary:

| # | Model | Why | Fits |
|---|-------|-----|------|
| 1 | gpt-oss:20b | fast reasoning model, pipeline shakedown | any GPU |
| 2 | gemma4:26b | newest Google open model | 48GB+ |
| 3 | deepseek-r1:32b | reasoning distill | 48GB+ |
| 4 | llama3.3:70b | Meta flagship at feasible size | 1× H100 |
| 5 | gpt-oss:120b | within-family scaling comparison | 1× H100 |
| 6 | glm-4.7-flash | GLM family (GLM-5 itself is too big) | 48GB+ |
| 7 | mistral-small3.2 | Mistral representative | any GPU |
| 8 | phi4:14b | strong small model | any GPU |
| 9–10 | llama3.1:8b, llama3.2:3b | small/tiny baselines | any GPU |
| 11 | qwen3.6:35b | optional strong extra | 48GB+ |
| 12 | minimax-m2.7 | stretch: ~230B MoE | 2× H100 |

GLM-5.x, DeepSeek-V4-Pro and Llama-405B don't fit the cluster — see
`not_feasible` in the config for the arithmetic.

## Repo layout

```
configs/models.json      model queue: tags, priorities, per-model SLURM resources
data/matches.json        fixtures (edit TBD teams as rounds resolve)
data/actuals.json        fill in real results after each match
src/predict.py           Ollama client: greedy+sampled generations, full logging
src/score.py             scoring + leaderboard (md/csv/json)
slurm/predict.slurm      generic one-model GPU job
slurm/run_queue.sh       submits the whole queue, chained one-after-one
scripts/setup_ollama.sh  user-space Ollama install (no root)
scripts/pull_model.sh    pre-download weights on the login node
tests/smoke_test.sh      end-to-end test with a mock server, no GPU needed
results/predictions/     aggregated predictions, one JSONL per model  (committed)
results/scores/          leaderboard outputs                          (committed)
logs/raw/                every raw model response                     (committed)
logs/slurm/              SLURM job + ollama server logs               (committed)
```

Everything needed to reproduce a number — model digest, Ollama version, seeds,
prompt hash, git commit, hostname, job ID — is embedded in every JSONL record.

A front-end dashboard reading `results/scores/leaderboard.json` is planned as a
follow-up project.
