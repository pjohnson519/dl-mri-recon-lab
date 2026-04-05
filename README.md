# DL MRI Reconstruction Lab — End-to-End VarNet

A teaching lab on learned MRI reconstruction using a simplified [End-to-End VarNet](https://arxiv.org/abs/2004.06688) applied to the [fastMRI](https://fastmri.org/) multicoil knee dataset. Self-contained — no fastMRI package dependency.

## Quick start

You will need **two terminal windows** on your laptop.

### Terminal 1 — SSH into the cluster and get a GPU

```bash
# Step 1: SSH into BigPurple
ssh YOUR_KID@bigpurple.nyumc.org

# Step 2: Clone the repo (first time only)
module load git
git clone https://github.com/pjohnson519/dl-mri-recon-lab.git
cd dl-mri-recon-lab

# Step 3: Get an interactive GPU node
# Option A — a100_dev (default for most students):
srun --partition=a100_dev --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty /bin/bash

# Option B — radiology partition (if you have access):
srun --partition=radiology --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty /bin/bash

# Step 4: Note which compute node you landed on
hostname
# e.g. a100-4003

# Step 5: Activate the environment and launch Jupyter
conda activate /gpfs/scratch/johnsp23/DLrecon_lab1/envs/varnet
cd ~/dl-mri-recon-lab
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0


Jupyter will print a URL like:

http://127.0.0.1:8888/?token=abc123def456...

Keep this terminal open — copy that URL.

### Terminal 2 — SSH tunnel from your laptop

Open a **new terminal on your laptop** (not on the cluster) and create an SSH tunnel:


ssh -N -L 8888:COMPUTE_NODE:8888 YOUR_KID@bigpurple.nyumc.org


Replace `COMPUTE_NODE` with the hostname from Step 4 (e.g. `a100-4003`).

### Open the notebook

Open a browser on your laptop and paste the URL from Jupyter (the one with the token).
Navigate to `notebooks/Lab1_VarNet.ipynb` and you're ready to go.

Run cells with **Shift+Enter**.

## Shared data (scratch)

All large files live on scratch — nothing needs to be downloaded:

| Path | Contents |
|------|----------|
| `/gpfs/scratch/johnsp23/DLrecon_lab1/data/knee/` | 855 multicoil knee H5 files (~814 GB) |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/data/fastMRI_paired_knee.csv` | Train/val/test split CSV |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/pretrained/varnet_random4x.pt` | 4x random model (12 cascades, 24 ch) |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/pretrained/varnet_random4x_noDC.pt` | 4x random, no data consistency |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/pretrained/varnet_random20x.pt` | 20x random model |
| `/gpfs/scratch/johnsp23/DLrecon_lab1/envs/varnet/` | Pre-built conda environment |

## Repository layout

```
dl-mri-recon-lab/
├── models/
│   ├── varnet.py         # SimpleVarNet, VarNetBlock, SensitivityModel
│   └── unet.py           # U-Net and NormUnet
├── utils/
│   ├── data.py           # FastMRIKneeDataset + DataLoader collate
│   ├── transforms.py     # FFT, k-space masking, complex math
│   └── metrics.py        # SSIM loss and metric
├── notebooks/
│   └── Lab1_VarNet.ipynb # <-- start here
├── scripts/
│   └── train.py          # Training script (for homework)
├── configs/
│   ├── random_4x.yaml
│   ├── random_4x_nodc.yaml
│   └── random_20x.yaml
└── environment.yml       # Conda env spec (if you need to recreate)
```

## Model overview

```
SimpleVarNet(num_cascades=12, chans=24, pools=4, use_dc=True)
│
├── SensitivityModel — estimates coil sensitivity maps from ACS lines
│     └── NormUnet(chans=8, pools=4)
│
└── 12 × VarNetBlock
      ├── NormUnet(chans=24, pools=4)    ← learned regularizer
      └── dc_weight (learnable scalar)   ← data consistency gate
```

**Forward pass:**
1. Estimate coil sensitivity maps from the center of k-space (ACS lines)
2. Each cascade: reduce to single-channel image → U-Net denoises → expand back to multi-coil k-space → soft data consistency step
3. IFFT → center-crop (640×368 → 320×320) → RSS coil combination

Setting `use_dc=False` freezes all DC weights at 0, converting the model to a pure image-processing pipeline.

## Lab notebook parts

| Part | Topic | What you do |
|------|-------|-------------|
| 0 | Setup & Data Exploration | Load k-space, visualize magnitude/phase/PDF, compare masks |
| 1 | Random 4x Inference | Run pretrained model with random and equispaced masks, measure SSIM |
| 2 | No Data Consistency | Ablate DC module, compare error patterns |
| 3 | 20x Acceleration | Extreme acceleration, examine pathology preservation |
| HW | Train Your Own Model | Train equispaced 4x model, compare to random-trained |

## Homework

**Option A:** Create `configs/equispaced_4x.yaml`, train with `scripts/train.py`, evaluate, and compare SSIM to the random-trained model on equispaced masks.

**Option B (Advanced):** Extend the model to 3D using slice-unique random masks or shifted equispaced masks.

See the notebook for detailed instructions and SLURM submission templates.

## Data format

Each H5 file contains one knee volume:
- `kspace`: `(slices, 15, 640, 368)` complex64 — 15-coil k-space
- `reconstruction_rss`: `(slices, 320, 320)` float32 — fully-sampled RSS target
- Attributes: `acquisition`, `max`, `norm`, `patient_id`
