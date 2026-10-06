# GDF-Restormer

GDF-Restormer is a global-coordinate-guided degradation-field restoration
network for blind metalens image restoration.

The restoration network receives the degraded RGB image together with
full-canvas global coordinates. The PSF used to synthesize the degradation is
not provided to the restoration network during training or inference.

## Quick Start

The recommended training entry is:

```bash
python train.py
```

Released model:

```text
Restormer_StrictGlobalDegField
```

Default registry name:

```text
StrictGlobalDegField_4L
```

Model size:

```text
25,971,477 trainable parameters
```

## Environment

Validated release environment:

```text
Python       3.10
PyTorch      2.0.1
torchvision  0.15.2
CUDA runtime 11.7
```

Create an environment:

```bash
conda create -n gdf-restormer python=3.10 pip -y
conda activate gdf-restormer
```

Install PyTorch:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2
```

Install the remaining default dependencies:

```bash
python -m pip install -r requirements.txt
```

Check the runtime:

```bash
python scripts/check_environment.py
```

Expected end:

```text
Model          : Restormer_StrictGlobalDegField
Parameters     : 25,971,477
[PASS] Minimal runtime and model construction are ready.
```

A Conda environment file is also provided:

```bash
conda env create -f environment.yml
conda activate gdf-restormer
```

## Dataset

Dataset repository:

```text
prpanda123/Metalens-HyperKvasir
```

Download:

```bash
bash scripts/download_dataset.sh
```

Expected structure:

```text
split_6500_1729/
├── train/
│   ├── gt/
│   └── meta/
├── val/
│   ├── gt/
│   └── meta/
└── split_manifest_fixed_sorted.csv
```

Fixed split:

```text
Train: 6500 pairs
Val  : 1729 pairs
```

For an existing local dataset:

```bash
python train.py \
  --data-root /path/to/split_6500_1729
```

## Preflight Check

```bash
python train.py \
  --data-root /path/to/split_6500_1729 \
  --dry-run
```

## One-Epoch Smoke Test

```bash
DATA=/path/to/split_6500_1729
bash scripts/smoke_test.sh "$DATA"
```

The smoke test covers:

```text
dataset loading
global-coordinate construction
model forward
loss
backward
optimizer step
validation
checkpoint save/load
full-image tiled evaluation
```

## Full Training

```bash
python train.py \
  --data-root /path/to/split_6500_1729
```

Default configuration:

```text
Epochs        : 100
Batch size    : 1
Initial LR    : 2e-4
Minimum LR    : 1e-6
Patch size    : 256
Loss          : L1 + 0.2 * SSIMLoss
Validation    : center validation every epoch
```

The default optimizer is AdamW.

## Main Files for GDF-Restormer

If you only want to reproduce the released GDF-Restormer, start with:

```text
train.py
config.yml
config/
engines/engine_strict_global_degfield.py
models/Res_Strict.py
models/Resbase.py
gdf_loss.py
data/dataset_GlobalMeta_FullCanvas.py
utils/
```

Main code path:

```text
train.py
   |
   +-- config/
   |
   +-- engines/engine_strict_global_degfield.py
   |      |
   |      +-- gdf_loss.py
   |      +-- utils/
   |      +-- data/dataset_GlobalMeta_FullCanvas.py
   |
   +-- models/Res_Strict.py
          |
          +-- models/Resbase.py
```

### Main model

```text
models/Res_Strict.py
```

Contains:

```text
Restormer_StrictGlobalDegField
Restormer_StrictDegField
STRICT_MODEL_REGISTRY
```

### Restormer primitives

```text
models/Resbase.py
```

### Training engine

```text
engines/engine_strict_global_degfield.py
```

### Dataset loader

```text
data/dataset_GlobalMeta_FullCanvas.py
```

### Loss

```text
gdf_loss.py
```

The released path imports:

```python
from gdf_loss import SSIMLoss
```

## Other Runnable Training Entries

The repository also contains additional runnable training scripts:

```text
train_strict_global_degfield.py
train_psf_degfield.py
train_global_psf_wavelet_moe.py
```

The default `requirements.txt` is intended for the released GDF-Restormer
training path. Other scripts or model implementations may require additional
packages.

## Repository Organization

```text
train.py                         recommended GDF-Restormer entry
config/                          configuration loader
data/                            dataset implementations
engines/                         training/evaluation engines
models/                          model implementations and baselines
loss/                            additional loss implementations
utils/                           utility functions
scripts/                         setup and reproducibility scripts
docs/                            repository documentation and audit material
tools/                           repository utility tools
results/                         retained reference results
```

The `models/` directory intentionally contains multiple restoration and
baseline architectures. They are not required by the default GDF-Restormer
training path; use the "Main Files for GDF-Restormer" section above if your
goal is only to reproduce the released method.

## Recommended Reproduction Sequence

```bash
python scripts/check_environment.py

python train.py \
  --data-root /path/to/split_6500_1729 \
  --dry-run

DATA=/path/to/split_6500_1729
bash scripts/smoke_test.sh "$DATA"

python train.py \
  --data-root /path/to/split_6500_1729
```

## License

See:

```text
LICENSE.md
THIRD_PARTY_NOTICES.md
```
