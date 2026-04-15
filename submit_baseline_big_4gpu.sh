#!/bin/bash
#SBATCH --job-name=vn_b_big
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/baseline_big_4x5x6x_4gpu_%j.out

source activate /gpfs/scratch/johnsp23/DLrecon_lab1/envs/varnet
cd ~/dl-mri-recon-lab

python -m torch.distributed.run --nproc_per_node=4 scripts/train.py \
    --config configs/random_4x5x6x_big.yaml \
    --data_path /gpfs/scratch/johnsp23/DLrecon_lab1/data/knee \
    --split_csv /gpfs/scratch/johnsp23/DLrecon_lab1/data/fastMRI_paired_knee.csv \
    --output_dir runs/baseline_big_4x5x6x
