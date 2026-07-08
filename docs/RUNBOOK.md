# RUNBOOK — reproducing everything on the SBU AI cluster

Every command below runs on the login node (`ai-slurm01`, reached via
`submit.ai.stonybrook.edu`) from the repo root — but the login node only ever
edits files and submits jobs. **All real work (Ollama install, weight downloads,
inference) happens inside SLURM jobs on compute nodes.** Downloads land on shared
`/home`, so they happen once and every node sees them.

The cluster has one SLURM partition (`defq`, 7-day limit) and six GPU nodes:

| Node | GPUs | VRAM each | Notes |
|------|------|-----------|-------|
| h100     | 4× H100      | 80GB | the big-model node (often fully booked) |
| quadro1-2| 8× RTX 8000  | 48GB | good default |
| tesla1-2 | 8× V100      | 32GB | fine for ≤20GB models |
| orion    | 4× V100      | 32GB | 1.5TB system RAM |

Cluster policy: no interactive notebooks; batch scripts that terminate — which is
exactly what we do. Verified 2026-07-08: compute nodes have outbound internet
(github.com and ollama.com reachable), so jobs can download for themselves.

## 0. One-time setup

```bash
cd ~/Explore
git clone git@github.com:enamcse/FIFA-World-Cup-2026-LLM-Predictions-QF-F.git
cd FIFA-World-Cup-2026-LLM-Predictions-QF-F

# Sanity-check the pipeline logic with a mock model server (pure-stdlib Python,
# runs in seconds, no GPU, no downloads - light enough for the login node; use
# sbatch --wrap 'bash tests/smoke_test.sh' if you prefer to keep it off entirely):
bash tests/smoke_test.sh        # must end with "SMOKE TEST PASSED"
```

There is no install step to run yourself: the first job that needs Ollama
downloads the **pinned version** (`OLLAMA_VERSION` in `scripts/env.sh`) onto
shared `/home` via `scripts/ensure_ollama.sh`, on the compute node, exactly once.

## 1. Prefetch weights + verify tags (submit this first)

Model names on ollama.com change; `configs/models.json` marks unverified tags
with `"tag_checked": false`. The prefetch job is CPU-only (holds no GPU), pulls
each tag on a compute node, and fails fast on wrong tags — costing nothing:

```bash
sbatch --export=ALL,MODELS="gpt-oss:20b gemma4:26b deepseek-r1:32b glm-4.7-flash mistral-small3.2 qwen3.6:35b phi4:14b llama3.1:8b llama3.2:3b" \
    slurm/prefetch.slurm
tail -f logs/slurm/wc26-prefetch-*.out    # watch; failed tags are listed with '!! FAILED'
```

If a tag fails, find the right one at https://ollama.com/search and fix it in
`configs/models.json`. Watch your quota: `du -sh ~/wc26_ollama_models` (~500GB cap).
GPU jobs also pull on their own if the prefetch hasn't run — prefetching just
saves GPU-allocated minutes and surfaces bad tags early.

## 2. Run one model (the learning loop)

```bash
sbatch --export=ALL,MODEL=gpt-oss:20b,STAGE=QF slurm/predict.slurm
```

What the job does, in order (read `slurm/predict.slurm`, it's short):
1. installs Ollama if missing (`scripts/ensure_ollama.sh`, lock-protected),
2. prints the GPUs SLURM gave it (`nvidia-smi`),
3. starts a private `ollama serve` on a job-unique port,
4. `ollama pull` (no-op if prefetched),
5. runs `src/predict.py`: for each match, 1 greedy + 10 sampled generations,
6. shuts the server down so the GPU is freed.

Monitor:

```bash
squeue -u $USER                          # queue state (PD=pending, R=running)
squeue -u $USER --start                  # scheduler's start-time estimate for pending jobs
tail -f logs/slurm/wc26-*-<jobid>.out    # live progress, one line per seed
tail -f logs/slurm/ollama-serve-<jobid>.log   # server side: load times, VRAM
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it ends
```

Outputs:
- `results/predictions/<model>.jsonl` — one aggregated prediction per match
- `logs/raw/<model>/<match>.jsonl` — every raw response, verbatim
- exit code 0 = every match produced a valid prediction

## 3. Run the whole queue

```bash
bash slurm/run_queue.sh --dry-run --stage QF   # inspect the sbatch commands first
bash slurm/run_queue.sh --stage QF             # submit for real (chained, one at a time)
bash slurm/run_queue.sh --stage QF --parallel  # deadline mode: all jobs schedulable at once
```

Default is chained (`--dependency=afterany`): polite, exactly one model on the
cluster at a time. `--parallel` drops the chaining so every job can backfill into
free GPUs independently — use it when a kickoff is close; the jobs are short
(1-2h limits) and mostly single-GPU, which is what SLURM's backfill scheduler
loves to squeeze in. Other variants:

```bash
bash slurm/run_queue.sh --only llama3.3:70b         # rerun one model
bash slurm/run_queue.sh --start-at 10               # only the multi-GPU tier
scancel -u $USER -n wc26-gemma4-26b                 # cancel one job by name
scancel -u $USER                                    # cancel everything of yours
```

**Deadline discipline:** predictions must exist before kickoff. QF1 is
July 9, 20:00 UTC. Priorities 1–9 are the fits-anywhere tier; 10–12 need
multiple 48GB GPUs and can wait for the queue to breathe.

## 4. Commit predictions BEFORE kickoff

The commit timestamp is your proof of pre-registration:

```bash
git add results/ logs/
git commit -m "Predictions for QF: <models run so far>"
git push
```

## 5. After each round: record results, advance the bracket

1. Edit `data/actuals.json` — fill `home_goals_90`, `away_goals_90` (score
   after 90 minutes, NOT after extra time), `advanced`, and a `source` URL.
2. Edit `data/matches.json` — replace `TBD` in SF1/SF2 (later TPP/F) with the
   real teams.
3. Re-run predictions for the newly-known matches only:
   ```bash
   bash slurm/run_queue.sh --stage SF     # later: --stage TPP,F
   ```
4. Commit before the next kickoff, as above.

## 6. Score the benchmark

```bash
python3 src/score.py          # instant, pure stdlib; skips matches with null actuals
git add results/scores data/actuals.json
git commit -m "Scores after <round>" && git push
```

Writes `results/scores/leaderboard.{md,csv,json}` (the JSON also contains the
per-match breakdown — that's the file the future dashboard should read) and
prints the table. Scoring rules are documented in the README and in
`src/score.py`'s docstring.

## Reproducibility guarantees

- **Ollama version is pinned** (`OLLAMA_VERSION` in `scripts/env.sh`) and
  recorded in every run's metadata.
- **Seeds are fixed** (greedy=42, samples 1001–1010) and recorded per generation.
- Every JSONL record embeds the **model digest, Ollama version, prompt SHA-256,
  prompt_version, git commit, hostname, and SLURM job ID**.
- Raw responses (including token counts and timings — useful for a
  tokens/sec comparison later) are never overwritten: files are append-only,
  and `score.py` uses the latest record per (model, match).
- Note: even with a fixed seed, GPU inference isn't guaranteed bit-identical
  across different GPU types/driver versions. To re-verify a result exactly,
  rerun on the same node type and compare digests.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `sbatch` job dies immediately, no output file | You submitted from outside the repo root — `logs/slurm/` didn't exist there. `cd` to repo root; `run_queue.sh` does this automatically. |
| `ollama pull` fails / tag not found | Wrong tag (fails in seconds): check https://ollama.com/search and fix `configs/models.json`. Transient network errors are retried 3× automatically. |
| A `:cloud` tag (e.g. glm-5.2:cloud) | Not runnable here — cloud tags have no downloadable weights; they execute on Ollama's servers. Out of scope. |
| Ollama log shows CUDA out-of-memory / model loads partly on CPU (very slow) | Give the job more/bigger GPUs: bump `gres` and add `"exclude": "tesla1,tesla2,orion"` (keeps it off 32GB V100s) in the config. Ollama splits across all visible GPUs automatically. |
| Job killed with `oom-kill` (system RAM, not VRAM) | Raise `mem` for that model in `configs/models.json` — Ollama mmaps weights and the page cache counts against the cgroup. |
| Job pending forever behind multi-day jobs | Keep `time` limits honest and short — backfill only slots you in if your job fits the gap. Check estimates with `squeue -u $USER --start`. `--parallel` helps: any free GPU can take any of your jobs. |
| Model returns unparseable output | Already handled: 3 retries + fallback JSON extraction; parse failures are visible in `logs/raw/.../*.jsonl` (`"parse_ok": false`). |
| Stale install lock (`.install-lock`) after a crashed job | `rmdir ~/wc26-tools/.install-lock` |
| Two jobs on one node fight over port 11434 | Can't happen: each job derives its port from its job ID. |

## Cleanup (be a good citizen)

Weights add up fast (gpt-oss:120b ≈ 65GB, llama3.3:70b ≈ 43GB; quota ~500GB).
`ollama list`/`rm` need a running server, so route them through a short CPU-only
job like everything else (checking disk usage is just a filesystem read — fine
anywhere):

```bash
du -sh ~/wc26_ollama_models        # quota check, safe on the login node

# list what's on disk / delete a model, on a compute node:
sbatch -p defq -c 2 --mem=8G -t 00:10:00 --output=logs/slurm/cleanup-%j.out --wrap '
  source scripts/env.sh; export CUDA_VISIBLE_DEVICES=""
  export OLLAMA_HOST=127.0.0.1:$(( 20000 + SLURM_JOB_ID % 10000 ))
  "$OLLAMA_BIN" serve & sleep 5
  "$OLLAMA_BIN" list
  # "$OLLAMA_BIN" rm <tag>
  kill %1'

rm -rf ~/wc26_ollama_models        # nuclear option when the project is over
```
