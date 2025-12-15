#!/bin/bash
#SBATCH --job-name=hc_v6_gpu
#SBATCH --output=reports/logs/v6_gpu_%j.out
#SBATCH --error=reports/logs/v6_gpu_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1         # Request 1 GPU
#SBATCH --cpus-per-task=8    # Request 8 CPU cores (enough to feed the GPU)
#SBATCH --mem=64G

set -e

TAG="bureau_v6"
SLUG="bureau-v6"
USER_ID="joshuarhb"

echo "🚀 Starting Full Hybrid Training (GPU Mode)..."
echo "   Node: $(hostname)"
echo "   GPU: $(nvidia-smi -L)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate homecredit
export PYTHONPATH=$PYTHONPATH:.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 1. RUN PYTHON TRAINING with --device gpu
python src/train_full_v6.py --device gpu --tag $TAG

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
    kaggle datasets version -p $UPLOAD_DIR -m "Full Hybrid v6 (GPU)"
else
    kaggle datasets create -p $UPLOAD_DIR
fi

# 3. PUSH KERNEL
echo "🚀 Pushing Submission Kernel..."
KERNEL_DIR="deployment/deploy_$TAG"
mkdir -p $KERNEL_DIR

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
echo "🎉 DONE! Track here: https://www.kaggle.com/code/$USER_ID/home-credit-submit-$SLUG"