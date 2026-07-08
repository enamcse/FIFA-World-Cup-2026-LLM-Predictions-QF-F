#!/bin/bash
# Ensure the pinned Ollama build is installed. Called INSIDE compute-node jobs
# (predict.slurm, prefetch.slurm) - do not run on the login node.
#
# The install lands on shared /home, so whichever job gets there first does the
# download once; later jobs see the binary and return immediately. A directory
# lock keeps concurrent jobs from downloading twice.
set -euo pipefail
source "$(dirname "$0")/env.sh"

[ -x "$OLLAMA_BIN" ] && { echo "ollama $("$OLLAMA_BIN" --version 2>/dev/null) present"; exit 0; }

LOCK="$WC26_TOOLS/.install-lock"
mkdir -p "$WC26_TOOLS"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another job is installing; waiting up to 10 min..."
    for _ in $(seq 1 60); do
        [ -x "$OLLAMA_BIN" ] && { echo "install appeared; done"; exit 0; }
        sleep 10
    done
    echo "FATAL: waited but $OLLAMA_BIN never appeared (stale lock? rmdir $LOCK)"
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

URL="https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION/ollama-linux-amd64.tar.zst"
TARBALL="$WC26_TOOLS/ollama-$OLLAMA_VERSION.tar.zst"
DEST="$WC26_TOOLS/ollama-$OLLAMA_VERSION"

echo "downloading ollama $OLLAMA_VERSION on $(hostname)..."
curl -fL --retry 3 -o "$TARBALL" "$URL"
mkdir -p "$DEST"
if tar --zstd -xf "$TARBALL" -C "$DEST" 2>/dev/null; then
    :
elif command -v zstd >/dev/null; then
    zstd -dc "$TARBALL" | tar -x -C "$DEST"
else
    echo "FATAL: neither 'tar --zstd' nor 'zstd' available on $(hostname)"
    exit 1
fi
rm -f "$TARBALL"
mkdir -p "$OLLAMA_MODELS"
echo "installed: $("$OLLAMA_BIN" --version)"
