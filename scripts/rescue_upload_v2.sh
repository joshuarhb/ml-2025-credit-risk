#!/bin/bash

# TAG = The folder name on your HPC (bureau_v2)
TAG=${1:-bureau_v2}

# SLUG = The sanitized name for Kaggle (bureau-v2)
# We replace underscores with hyphens automatically
SLUG=$(echo "$TAG" | tr '_' '-')

MODEL_DIR="./models/enhanced_$TAG"
UPLOAD_DIR="deployment/upload_$TAG"
KERNEL_DIR="deployment/deploy_$TAG"
USER_ID="joshuarhb" 

echo "🚑 Starting Rescue Upload for: $TAG"
echo "📦 Local Model: $MODEL_DIR"
echo "☁️  Target Kaggle Slug: $SLUG"

# 1. Prepare Upload Directory
mkdir -p $UPLOAD_DIR
cp $MODEL_DIR/* $UPLOAD_DIR/

# 2. Generate Metadata with VALID SLUG
# Note: "id" uses $SLUG, "title" can keep $TAG
cat > $UPLOAD_DIR/dataset-metadata.json <<EOF
{
  "title": "Home Credit Model $TAG",
  "id": "$USER_ID/home-credit-model-$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# 3. Create or Update Dataset
echo "⬆️ Uploading Dataset..."
if kaggle datasets status $USER_ID/home-credit-model-$SLUG > /dev/null 2>&1; then
    kaggle datasets version -p $UPLOAD_DIR -m "Rescue upload"
else
    kaggle datasets create -p $UPLOAD_DIR
fi

# 4. Prepare Kernel
echo "🚀 Preparing Submission Kernel..."
mkdir -p $KERNEL_DIR
cp src/inference.py $KERNEL_DIR/submit.py

# 5. Kernel Metadata (Pointing to the valid SLUG)
cat > $KERNEL_DIR/kernel-metadata.json <<EOF
{
  "id": "$USER_ID/home-credit-submit-$SLUG",
  "title": "Submit $TAG",
  "code_file": "submit.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "$USER_ID/home-credit-model-$SLUG",
    "home-credit-credit-risk-model-stability"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

# 6. Push Kernel
echo "⬆️ Pushing Kernel..."
kaggle kernels push -p $KERNEL_DIR

echo "✅ Rescue Complete! Check https://www.kaggle.com/code/$USER_ID/home-credit-submit-$SLUG"