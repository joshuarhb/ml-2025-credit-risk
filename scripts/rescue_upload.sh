#!/bin/bash

# TAG_LOCAL is the folder name on HPC (underscore is fine here)
TAG_LOCAL="advanced_v1"
# TAG_KAGGLE is the clean name for the URL (dashes only)
TAG_KAGGLE="advanced-v1"

echo "🚑 Starting Rescue Upload (Final Fix)..."

# 1. UPLOAD DATASET (Model)
# Your logs show the dataset was created successfully, but we'll prepare the folder anyway
UPLOAD_DIR="deployment/upload_rescue_$TAG_KAGGLE"
rm -rf $UPLOAD_DIR
mkdir -p $UPLOAD_DIR
cp ./models/enhanced_$TAG_LOCAL/* $UPLOAD_DIR/

cat > $UPLOAD_DIR/dataset-metadata.json <<EOF
{
  "title": "Home Credit Model $TAG_KAGGLE",
  "id": "joshuarhb/home-credit-model-$TAG_KAGGLE",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Try to create. If it exists (from your last run), we create a new version instead.
echo "☁️  Uploading/Updating Model..."
kaggle datasets create -p $UPLOAD_DIR || kaggle datasets version -p $UPLOAD_DIR -m "Update model"

# 2. PUSH KERNEL (Submission)
KERNEL_DIR="deployment/deploy_rescue_$TAG_KAGGLE"
rm -rf $KERNEL_DIR
mkdir -p $KERNEL_DIR
cp src/inference.py $KERNEL_DIR/submit.py

# FIX: Competition goes in competition_sources, Model goes in dataset_sources
cat > $KERNEL_DIR/kernel-metadata.json <<EOF
{
  "id": "joshuarhb/home-credit-submit-$TAG_KAGGLE",
  "title": "Submit $TAG_KAGGLE",
  "code_file": "submit.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "joshuarhb/home-credit-model-$TAG_KAGGLE"
  ],
  "competition_sources": [
    "home-credit-credit-risk-model-stability"
  ],
  "kernel_sources": []
}
EOF

echo "🚀 Pushing Kernel..."
kaggle kernels push -p $KERNEL_DIR

echo "✅ DONE! Check link: https://www.kaggle.com/code/joshuarhb/home-credit-submit-$TAG_KAGGLE"