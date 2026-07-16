#!/bin/bash
# End-to-end pipeline test with NO GPU and NO real Ollama:
# mock server -> predict.py -> fake actuals -> score.py -> leaderboard.
# Writes only to a temp dir; results/ and logs/ are untouched.
#   bash tests/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=12435
TMP=$(mktemp -d)
echo "workdir: $TMP"

python3 tests/mock_ollama.py "$PORT" &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true; rm -rf "$TMP"' EXIT
for _ in $(seq 1 20); do
    curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && break
    sleep 0.5
done

for MODEL in mock-alpha mock-beta; do
    python3 src/predict.py --model "$MODEL" --host "http://127.0.0.1:$PORT" \
        --stage QF --samples 5 \
        --outdir "$TMP/predictions" --rawdir "$TMP/raw" > "$TMP/$MODEL.out"
    echo "predict.py OK for $MODEL ($(wc -l < "$TMP/predictions/$MODEL.jsonl") matches)"
done

cat > "$TMP/actuals.json" <<'EOF'
{
  "QF1": {"home_goals_90": 2, "away_goals_90": 0, "advanced": "France", "source": "smoke"},
  "QF2": {"home_goals_90": 1, "away_goals_90": 1, "advanced": "Spain", "source": "smoke"},
  "QF3": {"home_goals_90": 0, "away_goals_90": 2, "advanced": "England", "source": "smoke"},
  "QF4": {"home_goals_90": 3, "away_goals_90": 1, "advanced": "Argentina", "source": "smoke"}
}
EOF

python3 src/score.py --predictions "$TMP/predictions" --actuals "$TMP/actuals.json" \
    --out "$TMP/scores"

python3 - "$TMP/scores/leaderboard.json" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
rows = doc["leaderboard"]
assert len(rows) == 2, f"expected 2 models, got {len(rows)}"
for r in rows:
    assert r["matches"] == 4, r
    assert 0 <= r["mean_rps"] <= 1 and 0 <= r["mean_brier"] <= 2, r
print("assertions passed")
EOF

echo
echo "SMOKE TEST PASSED"
