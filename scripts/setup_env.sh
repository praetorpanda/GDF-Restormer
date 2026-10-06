#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gdf-restormer}"

echo "[1/4] Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python=3.10 pip -y

# Make conda activation work inside the script.
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[2/4] Installing PyTorch 2.0.1 + torchvision 0.15.2 (CUDA 11.7 build)"
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2

echo "[3/4] Installing minimal project dependencies"
python -m pip install -r requirements.txt

echo "[4/4] Verifying environment"
python scripts/check_environment.py

echo
echo "Environment ready."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
