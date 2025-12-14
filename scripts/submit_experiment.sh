#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=home_credit_exp
#SBATCH --output=reports/logs/exp_%j.out   # Log stdout here
#SBATCH --error=reports/logs/exp_%j.err    # Log errors here
#SBATCH --time=01:00:00                    # 1 Hour is plenty for CSV training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16                 # Speed up Pandas/LightGBM
#SBATCH --mem=64G                          # Plenty of RAM to avoid "Killed"
#SBATCH --partition=main

# --- SETUP ---
echo "Job started on $(hostname) at $(date)"

# 1. Activate Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit
export PYTHONPATH=$PYTHONPATH:.

# 2. Fix C++ Library (Crucial for LightGBM)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- EXECUTION ---
# $1 is the argument we pass to sbatch (the experiment tag)
TAG=$1

if [ -z "$TAG" ]; then
    echo "❌ Error: No tag provided."
    echo "Usage: sbatch scripts/submit_experiment.sh your_tag_name"
    exit 1
fi

echo "🚀 Launching Experiment: $TAG"

# Run the automation script
bash scripts/run_experiment.sh $TAG

echo "Job finished at $(date)"