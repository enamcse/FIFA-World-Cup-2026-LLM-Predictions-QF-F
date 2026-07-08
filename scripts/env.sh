# Shared environment for all scripts/jobs. Source this; do not execute.
#   source scripts/env.sh
# Override any of these by exporting them before sourcing.

# Repo root (works when sourced from anywhere)
export WC26_ROOT="${WC26_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Where the user-space Ollama install lives (no root on the cluster)
export WC26_TOOLS="${WC26_TOOLS:-$HOME/wc26-tools}"
export OLLAMA_BIN="${OLLAMA_BIN:-$WC26_TOOLS/ollama/bin/ollama}"

# Model weights live on shared /home so login-node pulls are visible to compute nodes.
# NOTE: per-user quota is ~500GB; prune with '$OLLAMA_BIN rm <tag>' when done.
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/wc26_ollama_models}"

# Serve one request at a time, keep the model loaded for the whole job
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
