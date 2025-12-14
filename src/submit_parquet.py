import polars as pl
import pandas as pd
import lightgbm as lgb
import joblib
import os
import glob
import numpy as np
import gc

# --- CONFIGURATION ---
# Kaggle Paths
TEST_BASE_DIR = "/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/test"
MODEL_DIR = "/kaggle/input/home-credit-parquet-model"
MODEL_PATH = f"{MODEL_DIR}/lgbm_parquet.joblib"
WORKING_DIR = "/kaggle/working"

# Define the file groups (Must match training exactly)
BASE_F = "base.parquet"
DEPTH0_F = [
    ["static_0_0.parquet", "static_0_1.parquet"],
    ["static_cb_0.parquet"]
]
DEPTH1_F = [
    ["applprev_1_0.parquet", "applprev_1_1.parquet"],
    ["other_1.parquet"],
    ["deposit_1.parquet"],
    ["person_1.parquet"],
    ["debitcard_1.parquet"],
    ["tax_registry_a_1.parquet"],
    ["tax_registry_b_1.parquet"],
    ["tax_registry_c_1.parquet"],
    ["credit_bureau_a_1_0.parquet", "credit_bureau_a_1_1.parquet", "credit_bureau_a_1_2.parquet", "credit_bureau_a_1_3.parquet"],
    ["credit_bureau_b_1.parquet"],
    ["credit_bureau_b_2.parquet"],
    ["credit_bureau_a_2_0.parquet", "credit_bureau_a_2_1.parquet", "credit_bureau_a_2_2.parquet", "credit_bureau_a_2_3.parquet",
     "credit_bureau_a_2_4.parquet", "credit_bureau_a_2_5.parquet", "credit_bureau_a_2_6.parquet", "credit_bureau_a_2_7.parquet",
     "credit_bureau_a_2_8.parquet", "credit_bureau_a_2_9.parquet", "credit_bureau_a_2_10.parquet"],
    ["applprev_2.parquet"],
    ["person_2.parquet"]
]

# --- 1. COPY-PASTE PIPELINE FUNCTIONS ---

def set_table_dtypes(lf):
    for col, dtype in lf.collect_schema().items():
        if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
            lf = lf.with_columns(pl.col(col).cast(pl.Int64))
        elif col in ["date_decision"]:
            lf = lf.with_columns(pl.col(col).cast(pl.Date))
        elif col[-1] in ("P", "A"):
            lf = lf.with_columns(pl.col(col).cast(pl.Float64))
        elif col[-1] in ("D",):
            lf = lf.with_columns(pl.col(col).cast(pl.Date))
    return lf

def handle_dates(lf):
    cols = lf.collect_schema().names()
    if "date_decision" not in cols: return lf
    d_cols = [c for c in cols if c.endswith("D") and c != "date_decision"]
    if d_cols:
        lf = lf.with_columns([(pl.col(c) - pl.col("date_decision")).dt.total_days().alias(c) for c in d_cols])
    return lf.drop([c for c in ["date_decision", "MONTH"] if c in lf.collect_schema().names()])

def build_agg_exprs(tbl):
    cols = tbl.collect_schema().names()
    exprs = []
    exprs += [pl.col(c).max().alias(f"{c}_max") for c in cols if c.endswith(("A", "P", "M", "D", "T", "L"))]
    exprs += [pl.col(c).mean().alias(f"{c}_mean") for c in cols if c.endswith(("A", "P", "T", "L"))]
    exprs += [pl.col(c).min().alias(f"{c}_min") for c in cols if c.endswith("D")]
    return exprs

def process_group(group, depth=0, prefix="test"):
    paths = [os.path.join(TEST_BASE_DIR, f"{prefix}_{p}") for p in group]
    valid_paths = [p for p in paths if os.path.exists(p)]
    if not valid_paths: return None

    batches = []
    for p in valid_paths:
        lf = pl.scan_parquet(p)
        lf = set_table_dtypes(lf)
        if depth != 0:
            lf = lf.group_by("case_id").agg(build_agg_exprs(lf))
        batches.append(lf)
    
    if not batches: return None
    tbl = pl.concat(batches, how="vertical_relaxed")
    if len(batches) > 1: tbl = tbl.unique(subset=["case_id"])
    
    tbl = handle_dates(tbl)
    return tbl

# --- 2. EXECUTION ---

def run_submission():
    print("🚀 Starting Parquet Inference...")
    
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded.")

    # Process Test Data
    print("🔨 Aggregating Test Data...")
    agg_lfs = []
    
    for group in DEPTH0_F:
        res = process_group(group, depth=0, prefix="test")
        if res is not None: agg_lfs.append(res)
            
    for group in DEPTH1_F:
        res = process_group(group, depth=1, prefix="test")
        if res is not None: agg_lfs.append(res)

    # Load Base
    base_path = os.path.join(TEST_BASE_DIR, f"test_{BASE_F}")
    lf = pl.scan_parquet(base_path)
    
    # Join All
    for agg_lf in agg_lfs:
        lf = lf.join(agg_lf, on="case_id", how="left")
        
    # Collect
    print("💾 Collecting Data...")
    df_test = lf.collect()
    
    # 1. Extract Case IDs immediately (lightweight)
    case_ids = df_test["case_id"].to_numpy()
    
    # 2. Convert to Pandas (Heavy Operation 1)
    pdf = df_test.to_pandas()
    
    # 3. DELETE Polars object immediately to free RAM
    del df_test
    gc.collect()
    print("🧹 Polars DataFrame deleted.")

    # Align Columns
    expected_feats = model.feature_name()
    
    # Batch add missing columns (Fast & Low Memory)
    import pandas as pd
    missing_cols = list(set(expected_feats) - set(pdf.columns))
    if missing_cols:
        zeros_df = pd.DataFrame(0, index=pdf.index, columns=missing_cols)
        pdf = pd.concat([pdf, zeros_df], axis=1)
        del zeros_df
        gc.collect()

    # 4. Create Numpy Array (Heavy Operation 2)
    # Only keep the columns the model actually needs
    X = pdf[expected_feats].to_numpy()
    
    # 5. DELETE Pandas object immediately
    del pdf
    gc.collect()
    print("🧹 Pandas DataFrame deleted.")
    
    # Predict
    print("🔮 Predicting...")
    scores = model.predict(X)
    
    # 6. DELETE Numpy object
    del X
    gc.collect()

    # Save
    # Re-create a small Polars DF just for saving
    sub = pl.DataFrame({"case_id": case_ids, "score": scores})
    sub.write_csv("submission.csv")
    print("✅ submission.csv saved!")

if __name__ == "__main__":
    run_submission()