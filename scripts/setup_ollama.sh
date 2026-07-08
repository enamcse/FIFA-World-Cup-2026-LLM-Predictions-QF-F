#!/bin/bash
# One-time user-space Ollama install (no root). Run on the login node:
#   bash scripts/setup_ollama.sh
# Installs to $WC26_TOOLS/ollama (default ~/wc26-tools/ollama).
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [ -x "$OLLAMA_BIN" ]; then
    echo "Ollama already installed: $("$OLLAMA_BIN" --version 2>/dev/null || echo unknown)"
    echo "Delete $WC26_TOOLS/ollama and re-run to upgrade."
    exit 0
fi

mkdir -p "$WC26_TOOLS/ollama" "$OLLAMA_MODELS"
TGZ="$WC26_TOOLS/ollama-linux-amd64.tgz"

echo "Downloading Ollama (linux-amd64) from GitHub releases..."
curl -fL --retry 3 -o "$TGZ" \
    https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz

echo "Extracting to $WC26_TOOLS/ollama ..."
tar -xzf "$TGZ" -C "$WC26_TOOLS/ollama"
rm -f "$TGZ"

echo
echo "Installed: $("$OLLAMA_BIN" --version)"
echo "Model weights directory: $OLLAMA_MODELS"
echo "Next step: pre-pull a model with  bash scripts/pull_model.sh <tag>"
