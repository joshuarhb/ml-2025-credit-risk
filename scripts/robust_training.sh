#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=home_credit_robust
#SBATCH --output=./slurm_outputs/result_%j.txt
#SBATCH --error=./slurm_errors/error_%j.txt
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main

# --- COMMANDS ---
echo "Job started on $(hostname) at $(date)"

# 1. Activate Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit

# 2. THE FIX: Force Linux to use your Conda libraries first
# This fixes the 'GLIBCXX_3.4.30 not found' error
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Debug: LD_LIBRARY_PATH is now: $LD_LIBRARY_PATH"

# 3. Run the Model
python src/train.py

echo "Job finished at $(date)"