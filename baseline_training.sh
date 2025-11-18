#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=home_credit_base   # Name of the job
#SBATCH --output=./slurm_outputs/result_%j.txt        # Where 'print' output goes (%j = job ID)
#SBATCH --error=./slurm_errors/error_%j.txt          # Where error messages go
#SBATCH --time=00:30:00               # Time limit (30 mins is enough for this)
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores (Good for LightGBM)
#SBATCH --mem=32G                     # RAM (32GB is safe for this join)
#SBATCH --partition=main              # 'main' is standard on many clusters, check yours!

# --- COMMANDS ---
echo "Job started on $(hostname) at $(date)"

# 1. Load Python Module (This varies by university, try 'module avail python' to check)
# Common examples: 'module load python/3.9' or just 'module load python'
source ~/miniconda3/bin/activate

# 2. Activate your environment (If you created a virtual env)
# source my_env/bin/activate
conda activate homecredit

# 3. Run the script
python baseline_model.py

echo "Job finished at $(date)"