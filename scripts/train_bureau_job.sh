#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=home_credit_bureau
#SBATCH --output=reports/logs/bureau_%j.out
#SBATCH --error=reports/logs/bureau_%j.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=main

# Exit immediately if a command exits with a non-zero status
set -e

# --- 1. SETUP ENVIRONMENT ---
echo "🚀 Job started on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit
export PYTHONPATH=$PYTHONPATH:.

# Fix the C++ library issue for LightGBM
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Get the Experiment Tag (Default to 'bureau_v1')
TAG=${1:-bureau_v1}
MODEL_TAG="home-credit-model-$TAG"
USER_ID="joshuarhb" # CHANGE THIS IF YOUR KAGGLE USERNAME IS DIFFERENT

echo "🧪 Running Experiment: $TAG"

# --- 2. TRAINING ---
echo "🏋️ Starting Training Script (Enhanced)..."
python src/train_enhanced.py --tag $TAG

# Validation Check
if [ ! -f "./models/enhanced_$TAG/model.joblib" ]; then
    echo "❌ Training failed. Model file not found."
    exit 1
fi
echo "✅ Training Complete. Model saved."

# --- 3. DEPLOYMENT (Kaggle Upload) ---
echo "☁️ Preparing Kaggle Upload..."
UPLOAD_DIR="deployment/upload_$TAG"
mkdir -p $UPLOAD_DIR
cp ./models/enhanced_$TAG/* $UPLOAD_DIR/

# Check if dataset exists
echo "🔍 Checking if dataset $USER_ID/$MODEL_TAG exists..."
if kaggle datasets status $USER_ID/$MODEL_TAG > /dev/null 2>&1; then
    echo "🔄 Dataset exists. Creating new version..."
    # Generate Metadata (Needed even for version update)
    cat > $UPLOAD_DIR/dataset-metadata.json <<EOF
{
  "title": "Home Credit Model $TAG",
  "id": "$USER_ID/$MODEL_TAG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    kaggle datasets version -p $UPLOAD_DIR -m "Updated model version via Slurm"
else
    echo "🆕 Dataset does not exist. Creating..."
    # Generate Metadata
    cat > $UPLOAD_DIR/dataset-metadata.json <<EOF
{
  "title": "Home Credit Model $TAG",
  "id": "$USER_ID/$MODEL_TAG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    kaggle datasets create -p $UPLOAD_DIR
fi

# --- 4. SUBMISSION KERNEL ---
echo "🚀 Preparing Submission Kernel..."
KERNEL_DIR="deployment/deploy_$TAG"
mkdir -p $KERNEL_DIR

# Copy the MONOLITH inference script
cp src/inference.py $KERNEL_DIR/submit.py

# Create Kernel Metadata
cat > $KERNEL_DIR/kernel-metadata.json <<EOF
{
  "id": "$USER_ID/home-credit-submit-$TAG",
  "title": "Submit $TAG",
  "code_file": "submit.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "$USER_ID/$MODEL_TAG",
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
echo "Check your submission here: https://www.kaggle.com/code/$USER_ID/home-credit-submit-$TAG"