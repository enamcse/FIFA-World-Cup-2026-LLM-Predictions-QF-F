#!/bin/bash
# Download model weights on a compute node (CPU only, no GPU held) so GPU jobs
# start predicting immediately instead of downloading. Weights land on shared
# /home and are visible to every node. Submit FROM THE REPO ROOT:
#   sbatch --export=ALL,MODELS="gpt-oss:20b gemma4:26b phi4:14b" slurm/prefetch.slurm
#SBATCH --job-name=wc26-prefetch
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
echo "=== Prefetch job $SLURM_JOB_ID on $(hostname) at $(date -u +%FT%TZ) ==="
: "${MODELS:?export MODELS=\"tag1 tag2 ...\" when submitting}"

source scripts/env.sh
bash scripts/ensure_ollama.sh || exit 1

# No GPU was requested, so make absolutely sure we don't touch the node's GPUs
export CUDA_VISIBLE_DEVICES=""

PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
export OLLAMA_HOST="127.0.0.1:$PORT"
"$OLLAMA_BIN" serve > "logs/slurm/ollama-prefetch-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
    curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break
    sleep 2
done

RC=0
for MODEL in ${MODELS//,/ }; do
    echo "--- pulling $MODEL ---"
    OK=0
    for attempt in 1 2 3; do
        "$OLLAMA_BIN" pull "$MODEL" && { OK=1; break; }
        echo "pull attempt $attempt failed for $MODEL; retry in 30s"
        sleep 30
    done
    [ "$OK" = 1 ] || { echo "!! FAILED: $MODEL (bad tag? check ollama.com/search)"; RC=1; }
done

echo "--- on disk now ---"
"$OLLAMA_BIN" list
echo "disk usage: $(du -sh "$OLLAMA_MODELS" | cut -f1) of ~500GB quota"
echo "=== done at $(date -u +%FT%TZ), exit $RC ==="
exit $RC
