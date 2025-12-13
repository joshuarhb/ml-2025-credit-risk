import polars as pl
import pandas as pd
import joblib
import os
import gc

# --- CONFIGURATION ---
# 1. Competition Data
TEST_DIR = "/kaggle/input/home-credit-credit-risk-model-stability/csv_files/test"

# 2. YOUR NEW MODEL DATASET
# This must match the folder name Kaggle assigns to your new upload
MODEL_DIR = "/kaggle/input/hmoe-credit-robust-model"
MODEL_PATH = f"{MODEL_DIR}/lgbm_robust.joblib"
CAT_PATH = f"{MODEL_DIR}/cat_cols.joblib"

def run_submission():
    print("🚀 Starting Robust Inference...")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Available folders:", os.listdir("/kaggle/input"))
        return

    # 1. Load Resources
    model = joblib.load(MODEL_PATH)
    cat_cols = joblib.load(CAT_PATH)
    print("✅ Model & Categories Loaded.")

    # 2. Load Test Data
    # Handle the multi-part file logic again
    df_base = pl.read_csv(f"{TEST_DIR}/test_base.csv")
    df_static_0 = pl.read_csv(f"{TEST_DIR}/test_static_0_0.csv")
    
    if os.path.exists(f"{TEST_DIR}/test_static_0_1.csv"):
        df_static_1 = pl.read_csv(f"{TEST_DIR}/test_static_0_1.csv")
        df_static = pl.concat([df_static_0, df_static_1], how="vertical_relaxed")
    else:
        df_static = df_static_0
        
    df_test = df_base.join(df_static, on="case_id", how="left")
    
    # 3. Preprocess
    # Align columns with training
    pdf_test = df_test.to_pandas()
    
    # Explicitly ensure we have the columns the model expects
    # (LightGBM is picky about column order/existence)
    expected_features = model.feature_name_
    
    # Create a clean DataFrame with only expected features
    X_test = pd.DataFrame(index=pdf_test.index)
    
    for feat in expected_features:
        if feat in pdf_test.columns:
            X_test[feat] = pdf_test[feat]
        else:
            X_test[feat] = None # Fill missing features with NaN
            
    # Restore Categories
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')
            
    # Clean numeric objects
    num_cols = X_test.select_dtypes(include=['object']).columns
    for col in num_cols:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # 4. Predict
    scores = model.predict_proba(X_test)[:, 1]
    
    # 5. Save
    submission = pd.DataFrame({
        "case_id": df_test["case_id"],
        "score": scores
    })
    
    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv saved successfully!")

if __name__ == "__main__":
    run_submission()
