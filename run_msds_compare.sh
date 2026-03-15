#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

EPOCHS="${EPOCHS:-120}"
START_EPOCH="${START_EPOCH:-20}"
TS="$(date +%s)"
OUT_JSON="result/msds_compare_${TS}.json"
OUT_CSV="result/msds_compare_${TS}.csv"

latest_run_dir() {
  ls -1dt result/MSTGAD-MSDS-save-* 2>/dev/null | head -n 1 || true
}

run_once() {
  local tag="$1"
  shift
  echo "[run_msds_compare] Starting ${tag} run..." >&2
  python main.py --epochs "${EPOCHS}" "$@"
  local run_dir
  run_dir="$(latest_run_dir)"
  if [[ -z "${run_dir}" ]]; then
    echo "[run_msds_compare] ERROR: no result folder found after ${tag} run" >&2
    exit 1
  fi
  echo "[run_msds_compare] ${tag} run dir: ${run_dir}" >&2
  echo "${run_dir}"
}

# Baseline-like setting (disable new stabilization logic as much as possible).
BASE_DIR="$(run_once baseline \
  --eval_interval 1 \
  --train_eval_interval 1 \
  --threshold_search false \
  --monitor_warmup_epochs 9999 \
  --monitor_patience 9999 \
  --plateau_patience 9999)"

# Improved setting (new defaults from the implemented plan).
IMPROVED_DIR="$(run_once improved \
  --eval_interval 5 \
  --train_eval_interval 5 \
  --threshold_search true \
  --monitor_warmup_epochs 20 \
  --monitor_patience 6 \
  --monitor_min_delta 0.002 \
  --threshold_grid_step 0.01 \
  --threshold_ema 0.8 \
  --plateau_factor 0.5 \
  --plateau_patience 4)"

python compare_msds_logs.py \
  --baseline "${BASE_DIR}/running.log" \
  --improved "${IMPROVED_DIR}/running.log" \
  --start-epoch "${START_EPOCH}" \
  --out-json "${OUT_JSON}" \
  --out-csv "${OUT_CSV}"

echo "[run_msds_compare] baseline: ${BASE_DIR}/running.log"
echo "[run_msds_compare] improved: ${IMPROVED_DIR}/running.log"
echo "[run_msds_compare] summary json: ${OUT_JSON}"
echo "[run_msds_compare] summary csv: ${OUT_CSV}"
