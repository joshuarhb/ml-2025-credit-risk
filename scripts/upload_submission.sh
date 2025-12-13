#!/bin/bash

# --- CONFIGURATION ---
COMPETITION_NAME="home-credit-credit-risk-model-stability"
SUBMISSION_FILE="submission.csv"
MESSAGE="Baseline LightGBM Submission (via HPC)"

# --- CHECKS ---
if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "❌ Error: $SUBMISSION_FILE not found!"
    echo "   Run 'python creae_submission.py' first."
    exit 1
fi

# --- UPLOAD ---
echo "🚀 Uploading $SUBMISSION_FILE to Kaggle..."

# The actual command
kaggle competitions submit -c "$COMPETITION_NAME" -f "$SUBMISSION_FILE" -m "$MESSAGE"

# Check if it worked
if [ $? -eq 0 ]; then
    echo "✅ Upload Successful!"
    echo "   Check the leaderboard here:"
    echo "   https://www.kaggle.com/c/$COMPETITION_NAME/submissions"
else
    echo "❌ Upload Failed."
    echo "   Make sure you have ~/.kaggle/kaggle.json installed."
fi