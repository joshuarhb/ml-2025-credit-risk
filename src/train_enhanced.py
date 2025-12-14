import pandas as pd
import lightgbm as lgb
import joblib
import os
import argparse
import sys
from sklearn.metrics import roc_auc_score
from features_advanced import process_applprev_advanced # Import our new tool

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
    
    # 1. Load Base & Static (The standard stuff)
    print("⏳ Loading Static Data...")
    df_base = pd.read_csv(f"{DATA_DIR}/train_base.csv")
    
    # Load and concatenate static files
    static_files = [f for f in os.listdir(DATA_DIR) if "train_static_0" in f]
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in static_files]
    df_static = pd.concat(dfs, ignore_index=True)
    
    # 2. FEATURE ENGINEERING (The New Part)
    df_history = process_applprev_advanced(DATA_DIR)
    
    # 3. MERGE
    print("🔗 Joining Tables...")
    df_train = df_base.merge(df_static, on="case_id", how="left")
    df_train = df_train.merge(df_history, on="case_id", how="left")
    
    # Fill NaN for history (people with no history get 0)
    # ✅ NEW CORRECT CODE
    # Fill NaNs for the Advanced Features (total_apps, dpd_max, etc.)
    # We loop through them to be safe
    new_features = ["total_apps", "dpd_max", "dpd_mean", "amount_max", "is_refused_sum", "refusal_rate", "avg_loan_amount"]

    for col in new_features:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna(0)
    
    # 4. ROBUST SPLIT (Time-based)
    print("✂️ Splitting by Time...")
    df_train["date_decision"] = pd.to_datetime(df_train["date_decision"])
    
    # Sort by date to ensure strict past-vs-future split
    df_train = df_train.sort_values("date_decision")
    
    # Simple 80/20 Time Split
    split_idx = int(len(df_train) * 0.9)
    train = df_train.iloc[:split_idx]
    val = df_train.iloc[split_idx:]
    
    # Prepare X and y
    drop_cols = ["case_id", "target", "date_decision", "WEEK_NUM", "MONTH"]
    features = [c for c in train.columns if c not in drop_cols]
    
    X_train = train[features]
    y_train = train["target"]
    X_val = val[features]
    y_val = val["target"]
    
    # Handle Categories
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        X_train[c] = X_train[c].astype('category')
        X_val[c] = X_val[c].astype('category')
        
    print(f"🏋️ Training on {len(features)} features...")
    
    # 5. TRAIN (Robust Params)
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        colsample_bytree=0.5, # Robustness
        subsample=0.8,
        is_unbalance=True,    # Handle the imbalance
        metric="auc",
        n_jobs=16
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # 6. SAVE ARTIFACTS
    joblib.dump(model, f"{MODEL_DIR}/model.joblib")
    joblib.dump(features, f"{MODEL_DIR}/features.joblib")
    joblib.dump(cat_cols, f"{MODEL_DIR}/cat_cols.joblib")
    
    print(f"✅ Model Saved to {MODEL_DIR}")
    print(f"📊 Valid AUC: {model.best_score_['valid_0']['auc']:.4f}")

if __name__ == "__main__":
    run_training()