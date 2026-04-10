#!/bin/bash
#SBATCH --job-name=vn_equi4x
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/equi4x_%j.out

#conda activate /gpfs/scratch/johnsp23/DLrecon_lab1/envs/varnet
cd ~/dl-mri-recon-lab

python scripts/train.py --config configs/equispaced_4x.yaml --data_path /gpfs/scratch/johnsp23/DLrecon_lab1/data/knee --split_csv /gpfs/scratch/johnsp23/DLrecon_lab1/data/fastMRI_paired_knee.csv --output_dir runs/equi4x

