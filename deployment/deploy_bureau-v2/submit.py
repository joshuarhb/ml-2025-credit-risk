import pandas as pd
import joblib
import os
import glob
import gc
import numpy as np
import sys

# CONFIG - Kaggle Paths
TEST_DIR = "/kaggle/input/home-credit-credit-risk-model-stability/csv_files/test"
# Dynamic Model Directory Finding
try:
    # Looks for the dataset we uploaded via Slurm
    MODEL_DIR = [d for d in glob.glob("/kaggle/input/home-credit-model-*")][0]
except IndexError:
    MODEL_DIR = "/kaggle/input/home-credit-model-advanced-v1" # Fallback

MODEL_PATH = f"{MODEL_DIR}/model.joblib"
FEAT_PATH = f"{MODEL_DIR}/features.joblib"
CAT_PATH = f"{MODEL_DIR}/cat_cols.joblib"

# --- 1. BUREAU LOGIC (Embedded for Kaggle Safety) ---
def process_bureau_a_1(data_dir):
    """
    Aggregates Credit Bureau A (Depth 1) data.
    FIXED VERSION: Prevents Duplicate Column Names.
    """
    print("🏦 Engineering Credit Bureau A Features...")
    
    files = sorted(glob.glob(os.path.join(data_dir, "*credit_bureau_a_1_*.csv")))
    
    if not files:
        print("⚠️ No Credit Bureau A files found!")
        return pd.DataFrame(columns=["case_id"]).set_index("case_id")

    agg_dfs = []
    
    for f in files:
        print(f"   Processing file: {os.path.basename(f)}...")
        
        try:
            # Read header to create mapping
            header = pd.read_csv(f, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError:
            continue
            
        col_map = {}
        # 1. Map to unique names (preserve original suffix to ensure uniqueness)
        for c in header:
            if "outstandingdebt" in c: col_map[c] = f"bureau_debt_{c}"
            elif "monthlyinstlamount" in c: col_map[c] = f"bureau_annuity_{c}"
            elif "overdueamount" in c: col_map[c] = f"bureau_overdue_{c}"
            elif "dpd" in c and c.endswith("P"): col_map[c] = f"bureau_dpd_{c}"
        
        if not col_map:
            continue

        load_cols = ["case_id"] + list(col_map.keys())
        
        # 2. Process in Chunks
        for chunk in pd.read_csv(f, usecols=load_cols, chunksize=100000, low_memory=False):
            chunk = chunk.rename(columns=col_map)
            
            # Numeric Conversion
            bureau_cols = [c for c in chunk.columns if c.startswith("bureau_")]
            for c in bureau_cols:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce').fillna(0)

            # Define Aggregations
            aggs = {"case_id": "count"}
            for c in chunk.columns:
                if "bureau_debt" in c: aggs[c] = ["sum", "max"]
                elif "bureau_overdue" in c: aggs[c] = ["sum", "max"]
                elif "bureau_dpd" in c: aggs[c] = ["max", "mean"]
                elif "bureau_annuity" in c: aggs[c] = "sum"
            
            if "case_id" in aggs: del aggs["case_id"]
                
            agg_chunk = chunk.groupby("case_id").agg(aggs)
            
            # --- CRITICAL FIX HERE ---
            # Do NOT strip the suffix. Keep full name: bureau_debt_123A_sum
            new_cols = []
            for col_name, stat in agg_chunk.columns.values:
                new_cols.append(f"{col_name}_{stat}")
            
            agg_chunk.columns = new_cols
            # Add loan count manually
            agg_chunk["bureau_total_loans"] = chunk.groupby("case_id").size()
            
            agg_dfs.append(agg_chunk)
            del chunk
        gc.collect()

    # 3. Final Reduce
    print("   Combining Bureau chunks...")
    if not agg_dfs:
         return pd.DataFrame(columns=["case_id"]).set_index("case_id")
         
    full_df = pd.concat(agg_dfs)
    
    # 4. Dimensionality Reduction (Optional but recommended)
    # Since we have many split columns (debt_123A_sum, debt_456B_sum), 
    # we now aggregate them by case_id.
    
    # First, handle duplicates if any sneak in (Paranoia check)
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    
    # Generate final aggregation dict dynamically
    final_aggs = {}
    for c in full_df.columns:
        if "max" in c: final_aggs[c] = "max"
        elif "sum" in c: final_aggs[c] = "sum"
        elif "mean" in c: final_aggs[c] = "mean"
        elif "total_loans" in c: final_aggs[c] = "sum"
    
    final_df = full_df.groupby("case_id").agg(final_aggs)
    
    # 5. Simplify Features (Combine the disparate columns)
    # Instead of having 50 columns for debt, let's sum them into one 'total_debt'
    # This makes the model more robust and easier to interpret.
    
    print("   Simplifying Bureau Features...")
    # Find all columns related to Debt Sums
    debt_sum_cols = [c for c in final_df.columns if "bureau_debt" in c and "sum" in c]
    if debt_sum_cols:
        final_df["bureau_debt_total_sum"] = final_df[debt_sum_cols].sum(axis=1)
        # Drop the individual ones to save memory/noise
        final_df.drop(columns=debt_sum_cols, inplace=True)
        
    # Overdue Max
    overdue_max_cols = [c for c in final_df.columns if "bureau_overdue" in c and "max" in c]
    if overdue_max_cols:
        final_df["bureau_overdue_total_max"] = final_df[overdue_max_cols].max(axis=1)
        final_df.drop(columns=overdue_max_cols, inplace=True)

    # DPD Max
    dpd_max_cols = [c for c in final_df.columns if "bureau_dpd" in c and "max" in c]
    if dpd_max_cols:
        final_df["bureau_dpd_total_max"] = final_df[dpd_max_cols].max(axis=1)
        final_df.drop(columns=dpd_max_cols, inplace=True)

    return final_df

# --- 2. APPLPREV LOGIC ---
def process_applprev_advanced(data_dir):
    print("🧠 Engineering Advanced History Features...")
    files = [f for f in os.listdir(data_dir) if "applprev_1" in f and f.endswith(".csv")]
    agg_dfs = []
    
    for f in files:
        path = os.path.join(data_dir, f)
        try: header = pd.read_csv(path, nrows=0).columns.tolist()
        except: continue
        
        col_map = {}
        for c in header:
            if c.startswith("actualdpd"): col_map[c] = "dpd"
            elif c.startswith("credamount"): col_map[c] = "amount"
            elif c.startswith("status"): col_map[c] = "status"
        
        if not col_map: continue
        load_cols = ["case_id"] + list(col_map.keys())
        
        for chunk in pd.read_csv(path, usecols=load_cols, chunksize=50000):
            chunk = chunk.rename(columns=col_map)
            chunk["is_refused"] = 0
            if "status" in chunk.columns:
                chunk["is_refused"] = chunk["status"].astype(str).str.contains("D", na=False).astype(int)
            
            chunk["dpd"] = chunk["dpd"].fillna(0) if "dpd" in chunk.columns else 0
            chunk["amount"] = chunk["amount"].fillna(0) if "amount" in chunk.columns else 0

            agg = chunk.groupby("case_id").agg({
                "dpd": ["max", "mean"],
                "amount": ["max", "sum"],
                "is_refused": "sum",
                "case_id": "count"
            })
            agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
            agg.rename(columns={"case_id_count": "total_apps"}, inplace=True)
            agg_dfs.append(agg)
        gc.collect()

    if not agg_dfs: return pd.DataFrame(columns=["case_id"]).set_index("case_id")

    full_agg = pd.concat(agg_dfs)
    final_df = full_agg.groupby("case_id").agg({
        "dpd_max": "max", "dpd_mean": "mean", "amount_max": "max",
        "amount_sum": "sum", "is_refused_sum": "sum", "total_apps": "sum"
    })
    final_df["refusal_rate"] = final_df["is_refused_sum"] / final_df["total_apps"]
    final_df["avg_loan_amount"] = final_df["amount_sum"] / final_df["total_apps"]
    
    return final_df

# --- 3. MAIN PIPELINE ---
def run_inference():
    print("🚀 Starting Inference...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}"); return

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEAT_PATH)
    cat_cols = joblib.load(CAT_PATH)
    
    # Load Base
    df_base = pd.read_csv(f"{TEST_DIR}/test_base.csv")
    static_files = [f for f in os.listdir(TEST_DIR) if "test_static_0" in f]
    dfs = [pd.read_csv(os.path.join(TEST_DIR, f), low_memory=False) for f in static_files]
    if dfs: df_test = df_base.merge(pd.concat(dfs, ignore_index=True), on="case_id", how="left")
    else: df_test = df_base
    
    # Feature Engineering
    df_history = process_applprev_advanced(TEST_DIR)
    df_bureau = process_bureau_a_1(TEST_DIR)
    
    # Merge
    if not df_history.empty: df_test = df_test.merge(df_history, on="case_id", how="left")
    if not df_bureau.empty: df_test = df_test.merge(df_bureau, on="case_id", how="left")
    
    # Align Columns (Critical!)
    missing_cols = list(set(features) - set(df_test.columns))
    if missing_cols:
        zeros = pd.DataFrame(0, index=df_test.index, columns=missing_cols)
        df_test = pd.concat([df_test, zeros], axis=1)
            
    X_test = df_test[features]
    
    for c in cat_cols:
        if c in X_test.columns: X_test[c] = X_test[c].astype('category')
            
    print("🔮 Predicting...")
    scores = model.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({"case_id": df_test["case_id"], "score": scores})
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission Saved")

if __name__ == "__main__":
    run_inference()