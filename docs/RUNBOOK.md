# RUNBOOK — reproducing everything on the SBU AI cluster

Every command below runs on the login node (`ai-slurm01`, reached via
`submit.ai.stonybrook.edu`) from the repo root unless stated otherwise.
The cluster has one SLURM partition (`defq`, 7-day limit) and six GPU nodes:

| Node | GPUs | VRAM each | Notes |
|------|------|-----------|-------|
| h100     | 4× H100      | 80GB | the big-model node |
| quadro1-2| 8× RTX 8000  | 48GB | good default |
| tesla1-2 | 8× V100      | 32GB | fine for ≤20GB models |
| orion    | 4× V100      | 32GB | 1.5TB system RAM |

Storage is shared NFS `/home` (quota ~500GB/user), so weights pulled on the
login node are visible on every compute node. Cluster policy: no interactive
notebooks; batch scripts that terminate — which is exactly what we do.

## 0. One-time setup

```bash
git clone git@github.com:enamcse/FIFA-World-Cup-2026-LLM-Predictions-QF-F.git
cd FIFA-World-Cup-2026-LLM-Predictions-QF-F

# User-space Ollama install (no root; goes to ~/wc26-tools/ollama,
# weights to ~/wc26_ollama_models). Takes ~1 minute.
bash scripts/setup_ollama.sh

# Sanity-check the whole pipeline WITHOUT a GPU (mock model server):
bash tests/smoke_test.sh        # must end with "SMOKE TEST PASSED"
```

## 1. Verify model tags (5 minutes, do this once)

Model names on ollama.com change; `configs/models.json` marks unverified tags
with `"tag_checked": false`. A wrong tag fails instantly and costs nothing:

```bash
bash scripts/pull_model.sh gemma4:26b        # also pre-downloads the weights
```

If a tag 404s, search https://ollama.com/search?q=gemma for the right one and
fix it in `configs/models.json`. Pre-pulling the first few models on the login
node is recommended anyway — GPU jobs then start predicting immediately instead
of downloading. Watch your quota: `du -sh ~/wc26_ollama_models`.

## 2. Run one model (the learning loop)

```bash
sbatch --export=ALL,MODEL=gpt-oss:20b slurm/predict.slurm
```

What the job does, in order (read `slurm/predict.slurm`, it's short):
1. prints the GPUs SLURM gave it (`nvidia-smi`),
2. starts a private `ollama serve` on a job-unique port,
3. `ollama pull` (no-op if pre-pulled),
4. runs `src/predict.py`: for each match, 1 greedy + 10 sampled generations,
5. shuts the server down so the GPU is freed.

Monitor:

```bash
squeue -u $USER                          # queue state (PD=pending, R=running)
tail -f logs/slurm/wc26-*-<jobid>.out    # live progress, one line per seed
tail -f logs/slurm/ollama-serve-<jobid>.log   # server side: load times, VRAM
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it ends
```

Outputs:
- `results/predictions/<model>.jsonl` — one aggregated prediction per match
- `logs/raw/<model>/<match>.jsonl` — every raw response, verbatim
- exit code 0 = every match produced a valid prediction

## 3. Run the whole queue, one model after another

```bash
bash slurm/run_queue.sh --dry-run    # inspect the sbatch commands first
bash slurm/run_queue.sh              # submit for real
```

Jobs are chained with `--dependency=afterany`, so exactly one model runs at a
time, in priority order, each with the resources set in `configs/models.json`
(the 70B+ models are pinned to the H100 node). Useful variants:

```bash
bash slurm/run_queue.sh --stage QF                  # only quarter-finals
bash slurm/run_queue.sh --only llama3.3:70b         # rerun one model
bash slurm/run_queue.sh --start-at 5                # resume from priority 5
scancel -u $USER -n wc26-gemma4-26b                 # cancel one job by name
```

**Deadline discipline:** predictions must exist before kickoff. QF1 is
July 9, 20:00 UTC. If the queue won't finish in time, run `--stage QF` first
(4 matches × 11 generations is minutes per model, not hours) and commit.

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
python3 src/score.py          # any time; skips matches with null actuals
git add results/scores data/actuals.json
git commit -m "Scores after <round>" && git push
```

Writes `results/scores/leaderboard.{md,csv,json}` (the JSON also contains the
per-match breakdown — that's the file the future dashboard should read) and
prints the table. Scoring rules are documented in the README and in
`src/score.py`'s docstring.

## Reproducibility guarantees

- **Seeds are fixed** (greedy=42, samples 1001–1010) and recorded per generation.
- Every JSONL record embeds the **model digest, Ollama version, prompt SHA-256,
  prompt_version, git commit, hostname, and SLURM job ID**.
- Raw responses (including token counts and timings — useful for a
  tokens/sec comparison later) are never overwritten: files are append-only,
  and `score.py` uses the latest record per (model, match).
- Note: even with a fixed seed, GPU inference isn't guaranteed bit-identical
  across different GPU types/driver versions. To re-verify a result exactly,
  rerun on the same node type (`--nodelist=...`) and compare digests.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `sbatch` job dies immediately, no output file | You submitted from outside the repo root — `logs/slurm/` didn't exist there. `cd` to repo root; `run_queue.sh` does this automatically. |
| `pull` fails inside a job | Compute node may lack internet: pre-pull on the login node with `scripts/pull_model.sh`, then resubmit. |
| Ollama log shows CUDA out-of-memory / model loads partly on CPU (very slow) | Give the job more/bigger GPUs: `--nodelist=h100`, or `--gres=gpu:2` on quadro nodes. Ollama splits a model across all visible GPUs automatically. |
| Job killed with `oom-kill` (system RAM, not VRAM) | Raise `mem` for that model in `configs/models.json` — Ollama mmaps weights and the page cache counts against the cgroup. |
| Model returns unparseable output | Already handled: 3 retries + fallback JSON extraction; parse failures are visible in `logs/raw/.../*.jsonl` (`"parse_ok": false`). |
| Two jobs on one node fight over port 11434 | Can't happen: each job derives its port from its job ID. |

## Cleanup (be a good citizen)

```bash
~/wc26-tools/ollama/bin/ollama list                    # what's on disk
OLLAMA_MODELS=~/wc26_ollama_models ~/wc26-tools/ollama/bin/ollama rm <tag>
```

Remove weights for models you're done with — the 500GB quota fills fast
(gpt-oss:120b ≈ 65GB, llama3.3:70b ≈ 43GB).
