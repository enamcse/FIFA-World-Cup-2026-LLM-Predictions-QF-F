#!/bin/bash
# Generic one-model prediction job. Submit FROM THE REPO ROOT, e.g.:
#   sbatch --export=ALL,MODEL=gpt-oss:20b slurm/predict.slurm
# Optional exports: STAGE (default all), SAMPLES (default 10).
# Resource overrides at submit time, e.g. for a 70B model on two 48GB GPUs:
#   sbatch --gres=gpu:2 --exclude=tesla1,tesla2,orion --mem=96G \
#       --export=ALL,MODEL=llama3.3:70b slurm/predict.slurm
# slurm/run_queue.sh does all of this for you from configs/models.json.
# Everything (ollama binary, model weights) is downloaded ON the compute node.
#SBATCH --job-name=wc26
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date -u +%FT%TZ) ==="
: "${MODEL:?export MODEL=<ollama tag> when submitting}"
STAGE="${STAGE:-all}"
SAMPLES="${SAMPLES:-10}"

source scripts/env.sh
bash scripts/ensure_ollama.sh || { echo "FATAL: could not install ollama"; exit 1; }

# Unique port so concurrent jobs/users on a shared node never collide
PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
export OLLAMA_HOST="127.0.0.1:$PORT"

echo "--- GPU allocation ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

echo "--- Starting ollama serve on $OLLAMA_HOST (models: $OLLAMA_MODELS) ---"
mkdir -p logs/slurm "$OLLAMA_MODELS"
"$OLLAMA_BIN" serve > "logs/slurm/ollama-serve-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'echo "--- Stopping ollama (pid $SERVER_PID) ---"; kill $SERVER_PID 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 60); do
    if curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null; then READY=1; break; fi
    sleep 2
done
[ "$READY" = 1 ] || { echo "FATAL: ollama server did not come up; see logs/slurm/ollama-serve-$SLURM_JOB_ID.log"; exit 1; }
echo "Server ready: $(curl -s "http://$OLLAMA_HOST/api/version")"

echo "--- Ensuring model is present: $MODEL ---"
PULLED=0
for attempt in 1 2 3; do
    if "$OLLAMA_BIN" pull "$MODEL"; then PULLED=1; break; fi
    echo "pull attempt $attempt failed; retrying in 30s (prefetch weights with slurm/prefetch.slurm; check the tag on ollama.com/search)"
    sleep 30
done
[ "$PULLED" = 1 ] || { echo "FATAL: could not pull $MODEL"; exit 1; }
"$OLLAMA_BIN" show "$MODEL" || true

echo "--- Running predictions: stage=$STAGE samples=$SAMPLES ---"
python3 src/predict.py --model "$MODEL" --stage "$STAGE" --samples "$SAMPLES" \
    --host "http://$OLLAMA_HOST"
RC=$?

echo "--- GPU state after run ---"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true
echo "=== Finished at $(date -u +%FT%TZ) with exit code $RC ==="
exit $RC
