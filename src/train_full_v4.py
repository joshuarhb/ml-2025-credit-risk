import pandas as pd
import lightgbm as lgb
import joblib
import os
import argparse
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Import ALL your engineers
from features_advanced import process_applprev_advanced 
from src.features_bureau import process_bureau_a_1
from src.features_domain import process_domain_features
from src.features_financials import process_financial_features

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Training device")
parser.add_argument("--tag", type=str, default="bureau_v4", help="Experiment tag")
args = parser.parse_args()

# CONFIG
DATA_DIR = "./data/raw/csv_files/train"
TAG = args.tag
MODEL_DIR = f"./models/enhanced_{TAG}"
os.makedirs(MODEL_DIR, exist_ok=True)

def run_full_training():
    print(f"🚀 Starting Full Hybrid Training: {TAG} on {args.device.upper()}")
    
    # 1. LOAD & MERGE ALL DATA
    print("⏳ Loading Base & Static...")
    df_base = pd.read_csv(f"{DATA_DIR}/train_base.csv")
    static_files = [f for f in os.listdir(DATA_DIR) if "train_static_0" in f]
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False) for f in static_files]
    df_static = pd.concat(dfs, ignore_index=True)
    
    print("🛠 Processing History (ApplPrev)...")
    df_hist = process_applprev_advanced(DATA_DIR)
    
    print("🏦 Processing Bureau (External)...")
    df_bureau = process_bureau_a_1(DATA_DIR)
    
    print("🧠 Processing Domain (Trends)...")
    df_dom = process_domain_features(DATA_DIR)
    
    print("💰 Processing Financials (Tax/Debit)...")
    df_fin = process_financial_features(DATA_DIR)
    
    print("🔗 Merging Everything...")
    df_train = df_base.merge(df_static, on="case_id", how="left")
    df_train = df_train.merge(df_hist, on="case_id", how="left")
    df_train = df_train.merge(df_bureau, on="case_id", how="left")
    df_train = df_train.merge(df_dom, on="case_id", how="left")
    df_train = df_train.merge(df_fin, on="case_id", how="left")
    
    # 2. GLOBAL CLEANING
    print("🧹 Cleaning Data...")
    num_cols = df_train.select_dtypes(include=['number']).columns
    df_train[num_cols] = df_train[num_cols].fillna(0)
    
    # 3. TIME SPLIT
    df_train["date_decision"] = pd.to_datetime(df_train["date_decision"])
    df_train = df_train.sort_values("date_decision")
    split_idx = int(len(df_train) * 0.9)
    train = df_train.iloc[:split_idx]
    val = df_train.iloc[split_idx:]
    
    drop_cols = ["case_id", "target", "date_decision", "WEEK_NUM", "MONTH"]
    features = [c for c in train.columns if c not in drop_cols]
    
    X_train = train[features]
    y_train = train["target"]
    X_val = val[features]
    y_val = val["target"]
    
    # 4. CATEGORICAL HANDLING
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # 5. TRAIN LIGHTGBM (Keep CPU for safety/speed balance)
    print("⚡ Training LightGBM...")
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    for c in cat_cols:
        X_train_lgb[c] = X_train_lgb[c].astype('category')
        X_val_lgb[c] = X_val_lgb[c].astype('category')
        
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        colsample_bytree=0.5,
        subsample=0.8,
        is_unbalance=True,
        metric="auc",
        n_jobs=16, # CPU Cores
        verbose=-1
    )
    lgb_model.fit(
        X_train_lgb, y_train,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # 6. TRAIN CATBOOST (Use GPU if requested)
    print(f"🐱 Training CatBoost on {args.device.upper()}...")
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
    for c in cat_cols:
        X_train_cat[c] = X_train_cat[c].astype(str).fillna("Missing")
        X_val_cat[c] = X_val_cat[c].astype(str).fillna("Missing")

    cat_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "eval_metric": 'AUC',
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": 100
    }

    if args.device == "gpu":
        cat_params["task_type"] = "GPU"
        cat_params["devices"] = "0" # Use the first GPU allocated by Slurm
    else:
        cat_params["task_type"] = "CPU"

    cat_model = CatBoostClassifier(**cat_params)
    
    cat_model.fit(
        X_train_cat, y_train,
        eval_set=(X_val_cat, y_val),
        cat_features=cat_indices
    )
    
    # 7. SAVE EVERYTHING
    joblib.dump(lgb_model, f"{MODEL_DIR}/model.joblib")
    joblib.dump(cat_model, f"{MODEL_DIR}/cat_model.joblib")
    joblib.dump(features, f"{MODEL_DIR}/features.joblib")
    joblib.dump(cat_cols, f"{MODEL_DIR}/cat_cols.joblib")
    
    print(f"✅ Full Hybrid Model Saved to {MODEL_DIR}")

if __name__ == "__main__":
    run_full_training()