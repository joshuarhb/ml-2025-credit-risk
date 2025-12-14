#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=home_credit_final
#SBATCH --output=reports/logs/final_%j.out
#SBATCH --error=reports/logs/final_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=main

# --- 1. SETUP ENVIRONMENT ---
echo "🚀 Job started on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit
export PYTHONPATH=$PYTHONPATH:.

# Fix the C++ library issue for LightGBM
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Get the Experiment Tag (Default to 'advanced_v1' if empty)
TAG=${1:-advanced_v1}

echo "🧪 Running Experiment: $TAG"

# --- 2. TRAINING (Heavy Lifting) ---
echo "🏋️ Starting Training Script..."
python src/train_enhanced.py --tag $TAG

# Check if training succeeded
if [ ! -f "./models/enhanced_$TAG/model.joblib" ]; then
    echo "❌ Training failed. Model file not found."
    exit 1
fi
echo "✅ Training Complete. Model saved."

# --- 3. DEPLOYMENT (Kaggle Upload) ---
# Note: This runs on the compute node. If your HPC blocks internet on compute nodes,
# this part might fail. If so, you'll see a connection error in the logs.

echo "☁️ Preparing Kaggle Upload..."
UPLOAD_DIR="deployment/upload_$TAG"
mkdir -p $UPLOAD_DIR
cp ./models/enhanced_$TAG/* $UPLOAD_DIR/

# Initialize Dataset Metadata
cat > $UPLOAD_DIR/dataset-metadata.json <<EOF
{
  "title": "Home Credit Model $TAG",
  "id": "joshuarhb/home-credit-model-$TAG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Create Dataset
echo "⬆️ Uploading Model Dataset..."
kaggle datasets create -p $UPLOAD_DIR

# Prepare Submission Kernel
echo "🚀 Preparing Submission Kernel..."
KERNEL_DIR="deployment/deploy_$TAG"
mkdir -p $KERNEL_DIR
cp src/inference.py $KERNEL_DIR/submit.py

# Create Kernel Metadata
cat > $KERNEL_DIR/kernel-metadata.json <<EOF
{
  "id": "joshuarhb/home-credit-submit-$TAG",
  "title": "Submit $TAG",
  "code_file": "submit.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "joshuarhb/home-credit-model-$TAG",
    "home-credit-credit-risk-model-stability"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

# Push Kernel
echo "⬆️ Pushing Submission Kernel..."
kaggle kernels push -p $KERNEL_DIR

echo "🎉 PIPELINE FINISHED at $(date)"
echo "Check your submission here: https://www.kaggle.com/code/joshuarhb/home-credit-submit-$TAG"