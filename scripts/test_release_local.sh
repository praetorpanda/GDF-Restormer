#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage:"
  echo "  bash scripts/test_release_local.sh /path/to/split_6500_1729"
  exit 2
fi

echo "============================================================"
echo "[1/3] Python package consistency"
echo "============================================================"
python -m pip check

echo
echo "============================================================"
echo "[2/3] Minimal runtime / model construction"
echo "============================================================"
python scripts/check_environment.py

echo
echo "============================================================"
echo "[3/3] Dataset + config + model preflight"
echo "============================================================"
python train.py \
  --data-root "${DATA_ROOT}" \
  --dry-run

echo
echo "============================================================"
echo "LOCAL RELEASE PREFLIGHT PASSED"
echo "============================================================"
echo "Next optional step:"
echo "  bash scripts/smoke_test.sh \"${DATA_ROOT}\""
