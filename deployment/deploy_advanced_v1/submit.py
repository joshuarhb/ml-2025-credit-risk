import pandas as pd
import joblib
import os
import glob
import gc

# CONFIG - Kaggle Paths
TEST_DIR = "/kaggle/input/home-credit-credit-risk-model-stability/csv_files/test"
# We need to find the model folder dynamically or assume the user attached the right one
# Since our pipeline attaches "joshuarhb/home-credit-model-$TAG", we look for .joblib files
MODEL_DIR = [d for d in glob.glob("/kaggle/input/home-credit-model-*")][0]
MODEL_PATH = f"{MODEL_DIR}/model.joblib"
FEAT_PATH = f"{MODEL_DIR}/features.joblib"
CAT_PATH = f"{MODEL_DIR}/cat_cols.joblib"

def process_applprev_lite(data_dir):
    # SAME LOGIC AS TRAINING - Paste the function from Phase 1 here!
    # (For brevity, I assume you will copy-paste the process_applprev_lite function here)
    # ...
    return final_df

def run_inference():
    print("🚀 Starting Inference...")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEAT_PATH)
    cat_cols = joblib.load(CAT_PATH)
    
    # 1. Load Data
    df_base = pd.read_csv(f"{TEST_DIR}/test_base.csv")
    static_files = [f for f in os.listdir(TEST_DIR) if "test_static_0" in f]
    dfs = [pd.read_csv(os.path.join(TEST_DIR, f)) for f in static_files]
    df_static = pd.concat(dfs, ignore_index=True)
    
    # 2. Feature Engineering
    # IMPORTANT: You must copy the actual code of process_applprev_lite here
    # or import it if you upload the script as a utility script.
    # For now, simpler to just copy the function definition into this file.
    df_history = process_applprev_lite(TEST_DIR)
    
    # 3. Merge
    df_test = df_base.merge(df_static, on="case_id", how="left")
    df_test = df_test.merge(df_history, on="case_id", how="left")
    
    # 4. Align Columns
    for col in features:
        if col not in df_test.columns:
            df_test[col] = 0
            
    X_test = df_test[features]
    
    # Categories
    for c in cat_cols:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype('category')
            
    # 5. Predict
    scores = model.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({
        "case_id": df_test["case_id"],
        "score": scores
    })
    
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission Saved")

if __name__ == "__main__":
    run_inference()