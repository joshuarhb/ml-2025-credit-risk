#!/bin/bash
#SBATCH --job-name=home_credit_v4
#SBATCH --output=reports/logs/v4_%j.out
#SBATCH --error=reports/logs/v4_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=main

set -e

TAG="bureau_v4"
SLUG="bureau-v4"
USER_ID="joshuarhb"

echo "🚀 Starting Full Training V4..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit
export PYTHONPATH=$PYTHONPATH:.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 1. RUN PYTHON TRAINING
python src/train_full_v4.py

# 2. UPLOAD DATASET
echo "☁️  Uploading Artifacts..."
UPLOAD_DIR="deployment/upload_$TAG"
mkdir -p $UPLOAD_DIR
cp ./models/enhanced_$TAG/* $UPLOAD_DIR/

cat > $UPLOAD_DIR/dataset-metadata.json <<EOF
{
  "title": "Home Credit Model $TAG",
  "id": "$USER_ID/home-credit-model-$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

if kaggle datasets status $USER_ID/home-credit-model-$SLUG > /dev/null 2>&1; then
    kaggle datasets version -p $UPLOAD_DIR -m "Full Hybrid V4"
else
    kaggle datasets create -p $UPLOAD_DIR
fi

# 3. PUSH KERNEL
echo "🚀 Pushing Submission Kernel..."
KERNEL_DIR="deployment/deploy_$TAG"
mkdir -p $KERNEL_DIR

# COPY INFERENCE.PY (Ensure you updated it to include all feature functions!)
cp src/inference.py $KERNEL_DIR/submit.py

cat > $KERNEL_DIR/kernel-metadata.json <<EOF
{
  "id": "$USER_ID/home-credit-submit-$SLUG",
  "title": "Submit $TAG Full",
  "code_file": "submit.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "$USER_ID/home-credit-model-$SLUG"
  ],
  "competition_sources": [
    "home-credit-credit-risk-model-stability"
  ],
  "kernel_sources": []
}
EOF

kaggle kernels push -p $KERNEL_DIR
echo "🎉 DONE!"