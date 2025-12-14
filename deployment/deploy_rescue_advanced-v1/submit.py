import pandas as pd
import joblib
import os
import glob
import gc
import numpy as np

# CONFIG - Kaggle Paths
TEST_DIR = "/kaggle/input/home-credit-credit-risk-model-stability/csv_files/test"
# Find the model directory dynamically (looks for your uploaded dataset)
try:
    MODEL_DIR = [d for d in glob.glob("/kaggle/input/home-credit-model-*")][0]
except IndexError:
    # Fallback if specific tag not found
    MODEL_DIR = "/kaggle/input/home-credit-model-advanced-v1"

MODEL_PATH = f"{MODEL_DIR}/model.joblib"
FEAT_PATH = f"{MODEL_DIR}/features.joblib"
CAT_PATH = f"{MODEL_DIR}/cat_cols.joblib"

def process_applprev_advanced(data_dir):
    """
    Advanced Feature Engineering (Copy-pasted logic for Inference)
    """
    print("🧠 Engineering Advanced History Features (Inference)...")
    
    files = [f for f in os.listdir(data_dir) if "applprev_1" in f and f.endswith(".csv")]
    agg_dfs = []
    
    for f in files:
        path = os.path.join(data_dir, f)
        
        # 1. Dynamic Column Detection
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError:
            continue
            
        col_map = {}
        for c in header:
            if c.startswith("actualdpd"): col_map[c] = "dpd"
            elif c.startswith("credamount"): col_map[c] = "amount"
            elif c.startswith("status"): col_map[c] = "status"
            elif c.startswith("creationdate"): col_map[c] = "date"
        
        if not col_map:
            continue
            
        load_cols = ["case_id"] + list(col_map.keys())
        
        # 2. Process in Chunks
        # Smaller chunksize for inference to be safe
        for chunk in pd.read_csv(path, usecols=load_cols, chunksize=50000):
            chunk = chunk.rename(columns=col_map)
            
            # Logic
            if "status" in chunk.columns:
                chunk["is_refused"] = chunk["status"].astype(str).str.contains("D", na=False).astype(int)
            else:
                chunk["is_refused"] = 0
            
            if "dpd" in chunk.columns: chunk["dpd"] = chunk["dpd"].fillna(0)
            else: chunk["dpd"] = 0
                
            if "amount" in chunk.columns: chunk["amount"] = chunk["amount"].fillna(0)
            else: chunk["amount"] = 0

            # Aggregation
            agg = chunk.groupby("case_id").agg({
                "dpd": ["max", "mean"],
                "amount": ["max", "sum"],
                "is_refused": "sum",
                "case_id": "count"
            })
            
            agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
            agg.rename(columns={"case_id_count": "total_apps"}, inplace=True)
            agg_dfs.append(agg)
            del chunk
        gc.collect()

    if not agg_dfs:
        # If no history found (common in test set), return empty with correct index name
        return pd.DataFrame(columns=["case_id"]).set_index("case_id")

    full_agg = pd.concat(agg_dfs)
    
    final_df = full_agg.groupby("case_id").agg({
        "dpd_max": "max",
        "dpd_mean": "mean",
        "amount_max": "max",
        "amount_sum": "sum",
        "is_refused_sum": "sum",
        "total_apps": "sum"
    })
    
    # Ratios
    final_df["refusal_rate"] = final_df["is_refused_sum"] / final_df["total_apps"]
    final_df["avg_loan_amount"] = final_df["amount_sum"] / final_df["total_apps"]
    final_df["has_severe_default"] = (final_df["dpd_max"] > 90).astype(int)
    
    return final_df

def run_inference():
    print("🚀 Starting Inference...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        # List what IS there to help debug
        print(f"Contents of {MODEL_DIR}:")
        print(os.listdir(MODEL_DIR))
        return

    # Load artifacts
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEAT_PATH)
    cat_cols = joblib.load(CAT_PATH)
    print("✅ Model loaded.")
    
    # 1. Load Data
    df_base = pd.read_csv(f"{TEST_DIR}/test_base.csv")
    
    # Load static files
    static_files = [f for f in os.listdir(TEST_DIR) if "test_static_0" in f]
    dfs = []
    for f in static_files:
        dfs.append(pd.read_csv(os.path.join(TEST_DIR, f), low_memory=False))
    
    if dfs:
        df_static = pd.concat(dfs, ignore_index=True)
        df_test = df_base.merge(df_static, on="case_id", how="left")
    else:
        df_test = df_base
    
    # 2. Feature Engineering
    df_history = process_applprev_advanced(TEST_DIR)
    
    # 3. Merge
    if not df_history.empty:
        df_test = df_test.merge(df_history, on="case_id", how="left")
    
    # 4. Align Columns (Fill missing with 0)
    # We do this efficiently by only adding missing columns
    missing_cols = list(set(features) - set(df_test.columns))
    if missing_cols:
        zeros = pd.DataFrame(0, index=df_test.index, columns=missing_cols)
        df_test = pd.concat([df_test, zeros], axis=1)
            
    X_test = df_test[features]
    
    # Categories
    for c in cat_cols:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype('category')
            
    # 5. Predict
    print("🔮 Predicting...")
    scores = model.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({
        "case_id": df_test["case_id"],
        "score": scores
    })
    
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission Saved")

if __name__ == "__main__":
    run_inference()