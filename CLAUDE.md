# CLAUDE.md — DL MRI Reconstruction Lab

## What this project is

A teaching lab for NYU students on learned MRI reconstruction using a simplified End-to-End VarNet on the fastMRI multicoil knee dataset. Self-contained (no fastMRI package dependency).

- **Lab 1** (`notebooks/Lab1_VarNet.ipynb`): Data exploration, single-slice inference, DC ablation, extreme acceleration, equispaced-vs-random homework.
- **Lab 2** (`notebooks/Lab2_AdvancedVarNet.ipynb`): Multi-acceleration (4x/5x/6x), neighbouring-slice reconstruction, joint PD+PDFS contrast reconstruction, SSIM comparison across all variants.

## Environment

- **Cluster:** NYU BigPurple HPC (SLURM). GPU nodes via `srun` or `sbatch`.
- **Conda env:** `/gpfs/scratch/johnsp23/DLrecon_lab1/envs/varnet` (pre-built for students)
- **Python stack:** PyTorch, h5py, matplotlib, numpy, pyyaml
- **Activate with:** `conda activate /gpfs/scratch/johnsp23/DLrecon_lab1/envs/varnet`

## Key paths

| What | Path |
|------|------|
| Knee k-space H5s (~814 GB) | `/gpfs/scratch/johnsp23/DLrecon_lab1/data/knee/` |
| Train/val/test split CSV | `/gpfs/scratch/johnsp23/DLrecon_lab1/data/fastMRI_paired_knee.csv` |
| Pretrained checkpoints | `/gpfs/scratch/johnsp23/DLrecon_lab1/pretrained/` |
| Raw knee data (archive) | `/gpfs/data/lattanzilab/KneeRaw/` |

## Repo layout

```
models/varnet.py       # SimpleVarNet, VarNetBlock, SensitivityModel
models/unet.py         # U-Net and NormUnet
utils/data.py          # FastMRIKneeDataset, MultiSliceDataset, PairedContrastDataset
utils/transforms.py    # FFT, k-space masking, complex math
utils/metrics.py       # SSIM loss and metric
scripts/train.py       # Training script (all model variants)
scripts/eval.py        # Standalone evaluation (test/val SSIM + figures)
configs/*.yaml         # Training configs (mask type, cascades, lr, etc.)
notebooks/Lab1_VarNet.ipynb      # Lab 1: intro to VarNet (start here)
notebooks/Lab2_AdvancedVarNet.ipynb  # Lab 2: multi-slice, multi-contrast, multi-accel
submit_train.sh        # SLURM sbatch script for training
```

## Common tasks

### Run the notebook
```bash
srun --partition=a100_dev --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty /bin/bash
conda activate /gpfs/scratch/johnsp23/DLrecon_lab1/envs/varnet
cd ~/dl-mri-recon-lab
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

### Train a model
```bash
# Single-slice baseline:
sbatch submit_train.sh

# Multi-slice (3 adjacent slices, 4x/5x/6x):
sbatch submit_multislice.sh

# Joint contrast (PD+PDFS, 3 slices each, 4x/5x/6x):
sbatch submit_joint_contrast.sh

# Or directly:
python scripts/train.py --config configs/<config>.yaml \
    --data_path /gpfs/scratch/johnsp23/DLrecon_lab1/data/knee \
    --split_csv /gpfs/scratch/johnsp23/DLrecon_lab1/data/fastMRI_paired_knee.csv \
    --output_dir runs/<experiment_name>
```

## Conventions

- The notebook should stay **clean for students**: clear all outputs before committing (`jupyter nbconvert --clear-output --inplace`).
- Config files are YAML. Model hyperparams (cascades, channels, pools) and training params (lr, epochs, batch_size) live there, not hard-coded.
- H5 data format per volume: `kspace` (slices, 15, 640, 368) complex64, `reconstruction_rss` (slices, 320, 320) float32.
- The model center-crops from 640x368 to 320x320 before RSS coil combination.
- SSIM is the primary evaluation metric (both as loss and for reporting).
- Pretrained models use 12 cascades, 24 channels, 4 pools.
- Multi-acceleration configs list multiple `accelerations` / `center_fractions` pairs; the mask function randomly picks one per sample during training.
- Multi-slice models stack 3 adjacent slices along the coil dimension (15 coils x 3 = 45 channels) and reconstruct only the center slice.
- Joint-contrast models pair PD and PDFS volumes from the same exam using the `pd`/`pdfs` columns in the split CSV.

## Git

- `main` branch is the student-facing release.
- Don't commit large files, model weights, or data. Those live on scratch.
- Keep notebook outputs cleared in committed versions.
