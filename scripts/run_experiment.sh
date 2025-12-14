#!/bin/bash

# Usage: bash scripts/run_experiment.sh [TAG_NAME]
# Example: bash scripts/run_experiment.sh depth1_lite

TAG=$1
if [ -z "$TAG" ]; then
    echo "❌ Error: Please provide a tag name."
    echo "Usage: bash scripts/run_experiment.sh my_experiment_name"
    exit 1
fi

echo "========================================"
echo "🧪 STARTING EXPERIMENT: $TAG"
echo "========================================"

# 1. SETUP ENVIRONMENT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit
export PYTHONPATH=$PYTHONPATH:.
# FIX: Force Linux to use Conda's newer C++ libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 2. TRAIN ON HPC
# We run this directly (not sbatch) so we can wait for it to finish
# If it takes too long, you can wrap this in sbatch, but for CSVs it should be fast (<20m)
echo "🏋️ Training Model..."
python src/train_enhanced.py --tag $TAG

# Check if training succeeded
if [ ! -f "./models/enhanced_$TAG/model.joblib" ]; then
    echo "❌ Training failed. Model file not found."
    exit 1
fi

# 3. UPLOAD TO KAGGLE (The Model)
echo "☁️ Uploading Model to Kaggle..."
UPLOAD_DIR="deployment/upload_$TAG"
mkdir -p $UPLOAD_DIR
cp ./models/enhanced_$TAG/* $UPLOAD_DIR/

# Init Dataset
kaggle datasets init -p $UPLOAD_DIR
# Rewrite metadata with unique ID
cat > $UPLOAD_DIR/dataset-metadata.json <<EOF
{
  "title": "Home Credit Model $TAG",
  "id": "joshuarhb/home-credit-model-$TAG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

kaggle datasets create -p $UPLOAD_DIR

# 4. DEPLOY KERNEL (The Submission)
echo "🚀 Deploying Submission Kernel..."
KERNEL_DIR="deployment/deploy_$TAG"
mkdir -p $KERNEL_DIR

# Copy generic submission script
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

kaggle kernels push -p $KERNEL_DIR

echo "========================================"
echo "🎉 DONE! Experiment $TAG is live."
echo "Check progress: https://www.kaggle.com/code/joshuarhb/home-credit-submit-$TAG"
echo "========================================"