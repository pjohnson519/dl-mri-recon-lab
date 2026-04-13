# DL MRI Reconstruction Lab — End-to-End VarNet

A teaching lab on learned MRI reconstruction using a simplified [End-to-End VarNet](https://arxiv.org/abs/2004.06688) applied to the [fastMRI](https://fastmri.org/) multicoil knee dataset. Self-contained — no fastMRI package dependency.

## Quick start

We'll launch Jupyter on a GPU node through **NYU Open OnDemand** — no SSH tunnels required.

### 1. Pull the latest code

If you cloned this repo for Lab 1, you only have the Lab 1 version locally. Pull the Lab 2 updates first:

```bash
ssh YOUR_KID@bigpurple.nyumc.org
cd ~/dl-mri-recon-lab
git pull origin main
```

If you've never cloned the repo:
```bash
ssh YOUR_KID@bigpurple.nyumc.org
module load git
git clone https://github.com/pjohnson519/dl-mri-recon-lab.git
```

### 2. Launch a Jupyter session via Open OnDemand

1. In your browser, go to **https://ondemand.hpc.nyumc.org**
2. Log in with your KID
3. Start an **Interactive App → Jupyter** session with a GPU (see screenshots in lab handout)
4. Once the session is running, click **Connect to Jupyter**

### 3. Open the notebook

In the Jupyter file browser:
- Navigate to `dl-mri-recon-lab/notebooks/`
- Open `Lab2_AdvancedVarNet.ipynb` (Lab 1 is also there as a reference)

When prompted for a kernel, choose the `varnet` environment. If it's not listed, open a terminal in Jupyter and run:
```bash
conda activate /gpfs/scratch/johnsp23/DLrecon_lab1/envs/varnet
```

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
│   ├── data.py           # FastMRIKneeDataset, MultiSliceDataset, PairedContrastDataset
│   ├── transforms.py     # FFT, k-space masking, complex math
│   └── metrics.py        # SSIM loss and metric
├── notebooks/
│   ├── Lab1_VarNet.ipynb         # Lab 1: intro to VarNet
│   └── Lab2_AdvancedVarNet.ipynb # Lab 2: multi-slice, multi-contrast, multi-accel
├── scripts/
│   ├── train.py          # Training script (all model variants)
│   └── eval.py           # Standalone evaluation script
├── configs/              # YAML configs for each model variant
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

For multi-slice and joint-contrast variants, sensitivity maps are estimated per coil-group (one group per slice per contrast), and the cascade U-Net sees one image per group so it can learn cross-slice / cross-contrast features.

## Lab notebooks

### Lab 1 — Intro to VarNet (`Lab1_VarNet.ipynb`)

| Part | Topic |
|------|-------|
| 0 | Setup & data exploration |
| 1 | Random 4x inference |
| 2 | No data consistency ablation |
| 3 | 20x acceleration |
| HW | Train your own equispaced 4x model |

### Lab 2 — Advanced VarNet (`Lab2_AdvancedVarNet.ipynb`)

| Part | Topic |
|------|-------|
| 1 | Equispaced vs random mask training |
| 2 | Multi-acceleration (4x/5x/6x) masks |
| 3 | Neighbouring-slice reconstruction |
| 4 | Joint PD + PDFS contrast reconstruction |
| 5 | SSIM comparison across all models |

## Data format

Each H5 file contains one knee volume:
- `kspace`: `(slices, 15, 640, 368)` complex64 — 15-coil k-space
- `reconstruction_rss`: `(slices, 320, 320)` float32 — fully-sampled RSS target
- Attributes: `acquisition`, `max`, `norm`, `patient_id`
