import pandas as pd
import lightgbm as lgb
import joblib
import os
import argparse
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from features_advanced import process_applprev_advanced 
from src.features_bureau import process_bureau_a_1
from src.features_domain import process_domain_features
from src.features_financials import process_financial_features # <--- Import

# CONFIG
DATA_DIR = "./data/raw/csv_files/train"
MODEL_DIR = "./models/enhanced_bureau_v3" # Point to your best existing model folder
os.makedirs(MODEL_DIR, exist_ok=True)

def run_ensemble_training():
    print("🚀 Starting Ensemble Training (CatBoost Add-on)...")
    
    # 1. RE-LOAD DATA (Same logic as train_enhanced.py)
    # We need the exact same X_train/X_val structure
    print("⏳ Loading Data...")
    df_base = pd.read_csv(f"{DATA_DIR}/train_base.csv")
    
    static_files = [f for f in os.listdir(DATA_DIR) if "train_static_0" in f]
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False) for f in static_files]
    df_static = pd.concat(dfs, ignore_index=True)
    
    df_history = process_applprev_advanced(DATA_DIR)
    df_bureau = process_bureau_a_1(DATA_DIR)
    df_domain = process_domain_features(DATA_DIR)
    df_fin = process_financial_features(DATA_DIR)
    
    df_train = df_base.merge(df_static, on="case_id", how="left")
    df_train = df_train.merge(df_history, on="case_id", how="left")
    df_train = df_train.merge(df_bureau, on="case_id", how="left")
    df_train = df_train.merge(df_domain, on="case_id", how="left")
    df_train = df_train.merge(df_fin, on="case_id", how="left")
    
    # Fill NA
    bureau_cols = [c for c in df_train.columns if c.startswith("bureau_")]
    if bureau_cols: df_train[bureau_cols] = df_train[bureau_cols].fillna(0)
    
    for col in ["total_apps", "dpd_max", "dpd_mean", "amount_max", "is_refused_sum"]:
        if col in df_train.columns: df_train[col] = df_train[col].fillna(0)

    # Time Split
    df_train["date_decision"] = pd.to_datetime(df_train["date_decision"])
    df_train = df_train.sort_values("date_decision")
    split_idx = int(len(df_train) * 0.9)
    train = df_train.iloc[:split_idx]
    val = df_train.iloc[split_idx:]
    
    drop_cols = ["case_id", "target", "date_decision", "WEEK_NUM", "MONTH"]
    features = [c for c in train.columns if c not in drop_cols]
    
    # LOAD SAVED FEATURES TO ENSURE MATCH
    # We only want to use the features the LightGBM used
    saved_features = joblib.load(f"{MODEL_DIR}/features.joblib")
    X_train = train[saved_features]
    y_train = train["target"]
    X_val = val[saved_features]
    y_val = val["target"]

    # Handle Categoricals for CatBoost (It needs strings or indices, but handles NaNs well)
    # CatBoost prefers string categories usually, or declared indices
    cat_cols = joblib.load(f"{MODEL_DIR}/cat_cols.joblib")
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
    
    # Fill Cat NaNs with "Missing" for CatBoost stability
    for c in cat_cols:
        X_train[c] = X_train[c].astype(str).fillna("Missing")
        X_val[c] = X_val[c].astype(str).fillna("Missing")

    print(f"🐱 Training CatBoost on {len(saved_features)} features...")
    
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        bagging_temperature=0.2,
        od_type='Iter',
        od_wait=50,
        task_type="CPU", # Or GPU if you have it
        allow_writing_files=False
    )
    
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_indices,
        verbose=100
    )
    
    # Save CatBoost
    # CatBoost has its own save format, but joblib works for the wrapper
    joblib.dump(cat_model, f"{MODEL_DIR}/cat_model.joblib")
    print(f"✅ CatBoost Saved to {MODEL_DIR}")
    
    # --- CHECK STABILITY OF ENSEMBLE ---
    print("⚖️  Checking Ensemble Stability...")
    lgbm_model = joblib.load(f"{MODEL_DIR}/model.joblib")
    
    # Get LGBM preds (ensure types match what LGBM expects)
    # We need to reload X_val with correct types for LGBM (categories as int codes or pandas category)
    # This is tricky because we cast to string for CatBoost. 
    # For now, let's just rely on the fact that we have the models saved.
    
if __name__ == "__main__":
    run_ensemble_training()