#!/bin/bash
# Submit the whole model queue from configs/models.json as a chain of jobs
# that run strictly one after another (--dependency=afterany).
#
#   bash slurm/run_queue.sh                    # everything, priority order
#   bash slurm/run_queue.sh --stage QF         # only quarter-finals
#   bash slurm/run_queue.sh --only gpt-oss:20b,gemma4:26b
#   bash slurm/run_queue.sh --start-at 5       # skip priorities 1-4
#   bash slurm/run_queue.sh --dry-run          # print sbatch commands only
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, so #SBATCH --output paths work

STAGE="all"; SAMPLES="10"; ONLY=""; START_AT="1"; DRY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --stage)    STAGE="$2"; shift 2;;
        --samples)  SAMPLES="$2"; shift 2;;
        --only)     ONLY="$2"; shift 2;;
        --start-at) START_AT="$2"; shift 2;;
        --dry-run)  DRY=1; shift;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done

mkdir -p logs/slurm results/predictions logs/raw

# Emit one line per model: priority|tag|gres|nodelist|mem|time
PLAN=$(python3 - "$ONLY" "$START_AT" <<'EOF'
import json, sys
only = set(t for t in sys.argv[1].split(",") if t)
start_at = int(sys.argv[2])
cfg = json.load(open("configs/models.json"))
d = cfg["queue_defaults"]
for m in sorted(cfg["models"], key=lambda m: m["priority"]):
    if only and m["tag"] not in only:
        continue
    if not only and m["priority"] < start_at:
        continue
    print("|".join([str(m["priority"]), m["tag"],
                    m.get("gres", d["gres"]), m.get("nodelist", d["nodelist"]),
                    m.get("mem", d["mem"]), m.get("time", d["time"])]))
EOF
)
[ -n "$PLAN" ] || { echo "Nothing matched the filters."; exit 1; }

PREV=""
echo "priority  jobid    model                 resources"
while IFS='|' read -r PRIO TAG GRES NODELIST MEM TIME; do
    ARGS=(--partition=defq --gres="$GRES" --mem="$MEM" --time="$TIME"
          --job-name="wc26-$(echo "$TAG" | tr ':/' '--')"
          --export=ALL,MODEL="$TAG",STAGE="$STAGE",SAMPLES="$SAMPLES")
    [ -n "$NODELIST" ] && ARGS+=(--nodelist="$NODELIST")
    [ -n "$PREV" ]     && ARGS+=(--dependency=afterany:"$PREV")
    if [ -n "$DRY" ]; then
        echo "sbatch ${ARGS[*]} slurm/predict.slurm"
        PREV="<jobid-of-previous>"
        continue
    fi
    JOBID=$(sbatch --parsable "${ARGS[@]}" slurm/predict.slurm)
    printf "%-9s %-8s %-21s %s%s\n" "$PRIO" "$JOBID" "$TAG" "$GRES $MEM $TIME" \
        "${NODELIST:+ nodelist=$NODELIST}"
    PREV="$JOBID"
done <<< "$PLAN"

[ -n "$DRY" ] || echo
[ -n "$DRY" ] || echo "Queue submitted (chained with afterany). Watch with: squeue -u \$USER"
