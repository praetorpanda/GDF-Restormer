#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./split_6500_1729}"

python train.py   --config config_smoke.yml   --data-root "${DATA_ROOT}"   --name StrictGlobalDegField_4L_Smoke   --checkpoint-root ./checkpoints_smoke   --result-file ./results/smoke_result.txt   --max-eval-images 16
