import pandas as pd
import lightgbm as lgb
import joblib
import os
import argparse
from catboost import CatBoostClassifier

# Import Engineers
from features_advanced import process_applprev_advanced 
from src.features_bureau import process_bureau_a_1
from src.features_domain import process_domain_features
from src.features_financials import process_financial_features
from src.features_granular import process_granular_features # <--- Don't forget V5 Granular

# CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
parser.add_argument("--tag", type=str, default="bureau_v5", help="Tag")
args = parser.parse_args()

DATA_DIR = "./data/raw/csv_files/train"
MODEL_DIR = f"./models/enhanced_{args.tag}"
os.makedirs(MODEL_DIR, exist_ok=True)

def run_full_training():
    print(f"🚀 Starting V5 Final Hybrid Training on {args.device.upper()}")
    
    # 1. LOAD DATA
    df_base = pd.read_csv(f"{DATA_DIR}/train_base.csv")
    static_files = [f for f in os.listdir(DATA_DIR) if "train_static_0" in f]
    dfs = [pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False) for f in static_files]
    df_static = pd.concat(dfs, ignore_index=True)
    
    # 2. FEATURE ENGINEERING
    df_hist = process_applprev_advanced(DATA_DIR)
    df_bureau = process_bureau_a_1(DATA_DIR)
    df_dom = process_domain_features(DATA_DIR)
    df_fin = process_financial_features(DATA_DIR)
    df_gran = process_granular_features(DATA_DIR) # <--- V5 Feature
    
    # Merge
    df_train = df_base.merge(df_static, on="case_id", how="left")
    for df in [df_hist, df_bureau, df_dom, df_fin, df_gran]:
        df_train = df_train.merge(df, on="case_id", how="left")
        
    # Clean
    num_cols = df_train.select_dtypes(include=['number']).columns
    df_train[num_cols] = df_train[num_cols].fillna(0)
    
    # Time Split
    df_train["date_decision"] = pd.to_datetime(df_train["date_decision"])
    df_train = df_train.sort_values("date_decision")
    split_idx = int(len(df_train) * 0.9)
    train = df_train.iloc[:split_idx]
    val = df_train.iloc[split_idx:]
    
    features = [c for c in train.columns if c not in ["case_id", "target", "date_decision", "WEEK_NUM", "MONTH"]]
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    X_train = train[features]
    y_train = train["target"]
    X_val = val[features]
    y_val = val["target"]

    # --- NEW: FEATURE SELECTION (Scout) ---
    print("✂️ Running Feature Selection (Scout Model)...")
    # Train small model on subset
    scout_x = X_train.sample(n=100000, random_state=42)
    scout_y = y_train.loc[scout_x.index]
    
    for c in cat_cols: scout_x[c] = scout_x[c].astype('category')
    
    scout = lgb.LGBMClassifier(n_estimators=50, n_jobs=8)
    scout.fit(scout_x, scout_y)
    
    # Drop zero importance
    imp = pd.DataFrame({"f": features, "imp": scout.feature_importances_})
    useless = imp[imp["imp"] == 0]["f"].tolist()
    print(f"   Dropping {len(useless)} useless features.")
    
    # Apply Drop
    features = [f for f in features if f not in useless]
    X_train = X_train[features]
    X_val = X_val[features]
    cat_cols = [c for c in cat_cols if c in features]
    
    # --- FINAL TRAINING ---
    print("⚡ Training Final LightGBM...")
    # (Same training logic as v4, just using filtered features)
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    for c in cat_cols:
        X_train_lgb[c] = X_train_lgb[c].astype('category')
        X_val_lgb[c] = X_val_lgb[c].astype('category')
        
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=64,
        colsample_bytree=0.5, subsample=0.8, is_unbalance=True,
        metric="auc", n_jobs=16, verbose=-1
    )
    lgb_model.fit(X_train_lgb, y_train, eval_set=[(X_val_lgb, y_val)], 
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    
    print(f"🐱 Training Final CatBoost on {args.device.upper()}...")
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
    for c in cat_cols:
        X_train_cat[c] = X_train_cat[c].astype(str).fillna("Missing")
        X_val_cat[c] = X_val_cat[c].astype(str).fillna("Missing")

    cat_params = {
        "iterations": 1500, "learning_rate": 0.05, "depth": 6, "eval_metric": 'AUC',
        "random_seed": 42, "allow_writing_files": False, "verbose": 100,
        "task_type": "GPU" if args.device == "gpu" else "CPU"
    }
    if args.device == "gpu": cat_params["devices"] = "0"

    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train_cat, y_train, eval_set=(X_val_cat, y_val), cat_features=cat_indices)
    
    # Save
    joblib.dump(lgb_model, f"{MODEL_DIR}/model.joblib")
    joblib.dump(cat_model, f"{MODEL_DIR}/cat_model.joblib")
    joblib.dump(features, f"{MODEL_DIR}/features.joblib")
    joblib.dump(cat_cols, f"{MODEL_DIR}/cat_cols.joblib")
    print(f"✅ V5 Final Model Saved to {MODEL_DIR}")

if __name__ == "__main__":
    run_full_training()