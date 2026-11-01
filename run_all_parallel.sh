#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

API_KEY=""

declare -A PROJECTS
PROJECTS=(
  [pdfbox]="/Tmp/gamageyo/fika/experiments/pdfbox/pdfbox"
  [poitl]="/Tmp/gamageyo/fika/experiments/poi-tl/poi-tl"
  [graphhopper]="/Tmp/gamageyo/fika/experiments/graphhopper/core"
  [flink]="/Tmp/gamageyo/fika/experiments/flink/flink-core"
  [tika]="/Tmp/gamageyo/fika/experiments/tika/tika-core"
  [mybatis]="/Tmp/gamageyo/fika/experiments/mybatis-3"
  [tablesaw]="/Tmp/gamageyo/fika/experiments/tablesaw/json"
)

pids=()
for name in "${!PROJECTS[@]}"; do
  proj_dir="${PROJECTS[$name]}"
  echo "[$(date '+%H:%M:%S')] Starting $name ..."
  python3 -m junit_agent.main \
    "$proj_dir" \
    "$proj_dir/third_party_apis_full_methods.json" \
    --api openrouter \
    --log-file "agent_${name}.log" \
    --api-key "$API_KEY" \
    --iters 1 \
    --all \
    > "${name}.log" 2>&1 &
  pids+=("$!:$name")
done

echo ""
echo "All 7 jobs launched. Waiting for completion..."
echo ""

failed=0
for entry in "${pids[@]}"; do
  pid="${entry%%:*}"
  name="${entry##*:}"
  if wait "$pid"; then
    echo "[$(date '+%H:%M:%S')] ✓ $name finished successfully"
  else
    echo "[$(date '+%H:%M:%S')] ✗ $name FAILED (exit code $?)"
    failed=$((failed + 1))
  fi
done

echo ""
if [ "$failed" -eq 0 ]; then
  echo "All jobs completed successfully."
else
  echo "$failed job(s) failed. Check the .log files for details."
  exit 1
fi
