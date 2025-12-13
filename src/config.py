# src/config.py
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw/csv_files/train")
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw/csv_files/test")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports/figures")

# Hyperparameters
RANDOM_STATE = 42
SPLIT_WEEK = 78