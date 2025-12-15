#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=home_credit_ensemble
#SBATCH --output=reports/logs/ensemble_%j.out
#SBATCH --error=reports/logs/ensemble_%j.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=main

set -e # Stop on error

# --- CONFIG ---
SOURCE_TAG="bureau_v2"      # The LightGBM model we trust
TARGET_TAG="bureau_v4"       # New version
USER_ID="joshuarhb"
TARGET_SLUG="bureau-v3"     # Kaggle doesn't like underscores

echo "🚀 Starting Ensemble Pipeline on $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit
export PYTHONPATH=$PYTHONPATH:.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 1. PREPARE DIRECTORY (Clone v2 -> v3)
echo "📦 Cloning $SOURCE_TAG to $TARGET_TAG..."
mkdir -p ./models/enhanced_$TARGET_TAG
# Copy artifacts (Model, Features, Cols) so CatBoost uses exact same setup
cp ./models/enhanced_$SOURCE_TAG/*.joblib ./models/enhanced_$TARGET_TAG/

# 2. TRAIN CATBOOST
echo "🐱 Training CatBoost Add-on..."
# We run the python script. Ensure train_ensemble.py points to TARGET_TAG logic 
# (Or we pass the tag if you modified the script to accept args. 
# Since the script I gave you hardcoded 'bureau_v3', this works perfectly.)
python src/train_ensemble.py

# 3. KAGGLE DATASET UPLOAD
echo "☁️  Uploading Ensemble Artifacts..."
UPLOAD_DIR="deployment/upload_$TARGET_TAG"
mkdir -p $UPLOAD_DIR
cp ./models/enhanced_$TARGET_TAG/* $UPLOAD_DIR/

# Create Dataset Metadata
cat > $UPLOAD_DIR/dataset-metadata.json <<EOF
{
  "title": "Home Credit Model $TARGET_TAG",
  "id": "$USER_ID/home-credit-model-$TARGET_SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Upload
if kaggle datasets status $USER_ID/home-credit-model-$TARGET_SLUG > /dev/null 2>&1; then
    kaggle datasets version -p $UPLOAD_DIR -m "Ensemble Update"
else
    kaggle datasets create -p $UPLOAD_DIR
fi

# 4. KAGGLE KERNEL SUBMISSION
echo "🚀 Pushing Submission Kernel..."
KERNEL_DIR="deployment/deploy_$TARGET_TAG"
mkdir -p $KERNEL_DIR

# Copy your UPDATED inference.py (The Monolith)
cp src/inference.py $KERNEL_DIR/submit.py

# Create Kernel Metadata
cat > $KERNEL_DIR/kernel-metadata.json <<EOF
{
  "id": "$USER_ID/home-credit-submit-$TARGET_SLUG",
  "title": "Submit $TARGET_TAG Ensemble",
  "code_file": "submit.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "$USER_ID/home-credit-model-$TARGET_SLUG"
  ],
  "competition_sources": [
    "home-credit-credit-risk-model-stability"
  ],
  "kernel_sources": []
}
EOF

kaggle kernels push -p $KERNEL_DIR

echo "🎉 DONE! Track here: https://www.kaggle.com/code/$USER_ID/home-credit-submit-$TARGET_SLUG"