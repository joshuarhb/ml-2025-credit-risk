#!/bin/bash
#SBATCH --job-name=home_credit_parquet
#SBATCH --output=./reports/logs/parquet_%j.txt
#SBATCH --error=./reports/logs/parquet_err_%j.txt
#SBATCH --time=02:00:00        # Give it 2 hours, data processing takes time
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     # Polars LOVES cores. Give it more if you can (16-32).
#SBATCH --mem=64G              # Aggregation is memory hungry. 
#SBATCH --partition=main

echo "Job started on $(hostname) at $(date)"

# Activate Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit
export PYTHONPATH=$PYTHONPATH:.

# Fix the C++ library issue (Standard for LightGBM on HPC)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the new script
python src/train_parquet.py

echo "Job finished at $(date)"