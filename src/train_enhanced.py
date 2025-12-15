import pandas as pd
import lightgbm as lgb
import joblib
import os
import argparse
import sys
from sklearn.metrics import roc_auc_score
# Ensure these modules exist in your /src or root folder
from features_advanced import process_applprev_advanced 
from src.features_bureau import process_bureau_a_1

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="v1", help="Unique version tag for this run")
args = parser.parse_args()

# --- CONFIG ---
DATA_DIR = "./data/raw/csv_files/train"
MODEL_DIR = f"./models/enhanced_{args.tag}"
os.makedirs(MODEL_DIR, exist_ok=True)

def run_training():
    print(f"🚀 Starting Enhanced Training Run: {args.tag}")
    
    # 1. LOAD DATA
    print("⏳ Loading Base Data...")
    df_base = pd.read_csv(f"{DATA_DIR}/train_base.csv")

    print("⏳ Loading Static Data...")
    static_files = [f for f in os.listdir(DATA_DIR) if "train_static_0" in f]
        
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False) for f in static_files]
    df_static = pd.concat(dfs, ignore_index=True)

    # 2. FEATURE ENGINEERING
    print("🛠 Processing Internal History (ApplPrev)...")
    df_history = process_applprev_advanced(DATA_DIR)
    
    print("🛠 Processing External History (Bureau)...") 
    df_bureau = process_bureau_a_1(DATA_DIR)
    
    # 3. MERGE EVERYTHING
    print("🔗 Joining Tables...")
    df_train = df_base.merge(df_static, on="case_id", how="left")
    df_train = df_train.merge(df_history, on="case_id", how="left")
    df_train = df_train.merge(df_bureau, on="case_id", how="left") # <--- NEW MERGE
    
    # 4. CLEANING / IMPUTATION
    print("🧹 Handling Missing Values...")
    
    # Fill specific history features (Internal)
    history_features = ["total_apps", "dpd_max", "dpd_mean", "amount_max", 
                        "is_refused_sum", "refusal_rate", "avg_loan_amount"]
    for col in history_features:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna(0)

    # Fill bureau features (External)
    # Any column starting with 'bureau_' gets 0 if missing (implies no external record found)
    bureau_cols = [c for c in df_train.columns if c.startswith("bureau_")]
    if bureau_cols:
        df_train[bureau_cols] = df_train[bureau_cols].fillna(0)
    
    # 5. ROBUST SPLIT (Time-based)
    print("✂️ Splitting by Time...")
    df_train["date_decision"] = pd.to_datetime(df_train["date_decision"])
    
    # Sort by date to ensure strict past-vs-future split
    df_train = df_train.sort_values("date_decision")
    
    # Simple 90/10 Time Split (90% Train, 10% Valid)
    split_idx = int(len(df_train) * 0.9)
    train = df_train.iloc[:split_idx]
    val = df_train.iloc[split_idx:]
    
    # Define Features
    drop_cols = ["case_id", "target", "date_decision", "WEEK_NUM", "MONTH"]
    features = [c for c in train.columns if c not in drop_cols]
    
    X_train = train[features]
    y_train = train["target"]
    X_val = val[features]
    y_val = val["target"]
    
    # 6. SAFETY CHECK: Drop Constant Columns
    # If a feature has 0 or 1 unique value, it provides no information and breaks some algos.
    print("🗑 Checking for constant columns...")
    drop_candidates = [c for c in X_train.columns if X_train[c].nunique() <= 1]
    if drop_candidates:
        print(f"   Dropping {len(drop_candidates)} constant features")
        X_train = X_train.drop(columns=drop_candidates)
        X_val = X_val.drop(columns=drop_candidates)
        features = X_train.columns.tolist() # Update feature list
    
    # 7. CATEGORICAL HANDLING
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"   Found {len(cat_cols)} categorical features.")
    for c in cat_cols:
        X_train[c] = X_train[c].astype('category')
        X_val[c] = X_val[c].astype('category')
        
    print(f"🏋️ Training on {len(features)} final features...")
    
    # 8. TRAIN
    model = lgb.LGBMClassifier(
        n_estimators=1200,      # Slightly increased for more data
        learning_rate=0.03,
        num_leaves=64,
        colsample_bytree=0.5,   # Robustness against overfitting
        subsample=0.8,
        is_unbalance=True,      # Handle the default rate imbalance
        metric="auc",
        n_jobs=16,
        verbose=-1              # Reduce log noise
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # 9. SAVE ARTIFACTS
    joblib.dump(model, f"{MODEL_DIR}/model.joblib")
    joblib.dump(features, f"{MODEL_DIR}/features.joblib")
    joblib.dump(cat_cols, f"{MODEL_DIR}/cat_cols.joblib")
    
    print(f"✅ Model Saved to {MODEL_DIR}")
    print(f"📊 Valid AUC: {model.best_score_['valid_0']['auc']:.4f}")

if __name__ == "__main__":
    run_training()