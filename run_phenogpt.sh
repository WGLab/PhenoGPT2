#!/bin/bash
#SBATCH --job-name=phenogpt2_ft
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
module load CUDA/12.1.1

python phenogpt2_training.py
#####################sbatch -p gpuq run_phenogpt.sh################