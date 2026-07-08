#!/bin/bash
# Pre-pull a model's weights on the login node so GPU jobs don't burn GPU
# time downloading. Starts a throwaway CPU-only Ollama server on a random
# port, pulls, prints model info, and shuts the server down.
#   bash scripts/pull_model.sh gpt-oss:20b
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL="${1:?usage: pull_model.sh <ollama-model-tag>}"
[ -x "$OLLAMA_BIN" ] || { echo "Ollama not installed - run scripts/setup_ollama.sh first"; exit 1; }

PORT=$(( 21000 + RANDOM % 10000 ))
export OLLAMA_HOST="127.0.0.1:$PORT"
LOG="$(mktemp /tmp/ollama-pull-XXXX.log)"

"$OLLAMA_BIN" serve >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
    curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break
    sleep 1
done

echo "Pulling $MODEL into $OLLAMA_MODELS (this can take a while for big models)..."
"$OLLAMA_BIN" pull "$MODEL"
echo
"$OLLAMA_BIN" show "$MODEL" || true
echo
echo "On-disk models:"
"$OLLAMA_BIN" list
echo "Disk usage: $(du -sh "$OLLAMA_MODELS" | cut -f1)  (quota ~500GB; prune with '$OLLAMA_BIN rm <tag>')"
