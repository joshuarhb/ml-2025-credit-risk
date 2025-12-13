import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg') # Force headless backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc
import joblib
import os

# --- CONFIGURATION ---
DATA_DIR = "./data/original-data/csv_files/train" 
MODEL_PATH = "lgbm_robust.joblib"
# The competition has 91 weeks. We split roughly at 85% time.
# Weeks 0-78 = Train (approx 1.2M rows)
# Weeks 79-91 = Validation (approx 300k rows - The "Future")
SPLIT_WEEK = 78 
RANDOM_STATE = 42

def gini_stability(y_true, y_pred, weeks, w_fallingrate=88.0, w_resstd=-0.5):
    """
    Calculates the stability metric used in the competition.
    Metric = mean(gini) + w_fallingrate * min(0, slope) + w_resstd * std(residuals)
    """
    # 1. Calculate Gini per week
    # Gini = 2 * AUC - 1
    gini_in_time = []
    possible_weeks = sorted(weeks.unique())
    
    for w in possible_weeks:
        mask = (weeks == w)
        if mask.sum() < 10: continue # Skip weeks with too little data
        
        auc_w = roc_auc_score(y_true[mask], y_pred[mask])
        gini_w = 2 * auc_w - 1
        gini_in_time.append(gini_w)
    
    gini_in_time = np.array(gini_in_time)
    
    # 2. Linear Regression (Slope)
    x = np.arange(len(gini_in_time))
    slope, intercept, _, _, _ = linregress(x, gini_in_time)
    
    # 3. Penalties
    falling_penalty = min(0, slope) * w_fallingrate
    std_penalty = np.std(gini_in_time - (intercept + slope * x)) * w_resstd
    
    stability_score = np.mean(gini_in_time) + falling_penalty + std_penalty
    
    return stability_score, gini_in_time, possible_weeks, slope

def run_pipeline():
    print("⏳ Loading Data with Polars...")
    
    try:
        df_base = pl.read_csv(f"{DATA_DIR}/train_base.csv")
        df_static_0 = pl.read_csv(f"{DATA_DIR}/train_static_0_0.csv")
        df_static_1 = pl.read_csv(f"{DATA_DIR}/train_static_0_1.csv")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None, None

    # Join and Process
    df_static = pl.concat([df_static_0, df_static_1], how="vertical_relaxed")
    print("🔗 Joining Tables...")
    df_train = df_base.join(df_static, on="case_id", how="left")
    
    # Garbage collection
    del df_static_0, df_static_1, df_static
    gc.collect()
    
    print("🧹 Preprocessing...")
    
    # --- HANDLING WEEK_NUM ---
    # We explicitly NEED 'WEEK_NUM' for the split, so we select it here.
    # We will drop it from X_train later.
    
    # Filter columns with too many nulls
    null_counts = df_train.null_count()
    total_rows = df_train.height
    cols_to_keep = [col for col in df_train.columns if null_counts[col][0] / total_rows < 0.6]
    df_train = df_train.select(cols_to_keep)
    
    pdf_train = df_train.to_pandas()
    
    cat_cols = pdf_train.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        pdf_train[col] = pdf_train[col].astype('category')
        
    # --- TIME-BASED SPLIT ---
    print(f"✂️ Splitting Data by Time (Cutoff: Week {SPLIT_WEEK})...")
    
    mask_train = pdf_train['WEEK_NUM'] <= SPLIT_WEEK
    mask_val = pdf_train['WEEK_NUM'] > SPLIT_WEEK
    
    # Create Train/Val sets
    # IMPORTANT: We drop target, case_id, date_decision.
    # We ALSO drop WEEK_NUM from the features (X), but keep it for metrics (weeks_val)
    
    features_to_drop = ['target', 'case_id', 'date_decision', 'WEEK_NUM']
    
    X_train = pdf_train[mask_train].drop(columns=features_to_drop)
    y_train = pdf_train[mask_train]['target']
    
    X_val = pdf_train[mask_val].drop(columns=features_to_drop)
    y_val = pdf_train[mask_val]['target']
    
    # Keep validation weeks for the stability metric
    weeks_val = pdf_train[mask_val]['WEEK_NUM']
    
    print(f"   Train Shape: {X_train.shape}")
    print(f"   Val Shape:   {X_val.shape}")

    # --- MODEL TRAINING ---
    print(f"🏋️ Training LightGBM...")
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,          # Increased since we have more data and early stopping
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1 
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    print(f"💾 Saving model to '{MODEL_PATH}'...")
    joblib.dump(model, MODEL_PATH)
    
    print(f"💾 Saving category list...")
    joblib.dump(cat_cols, "cat_cols.joblib")
    
    # --- SCORING ---
    print("📊 Calculating Stability Metrics...")
    val_preds = model.predict_proba(X_val)[:, 1]
    
    # 1. Standard AUC
    auc_score = roc_auc_score(y_val, val_preds)
    
    # 2. Custom Stability Score
    stab_score, gini_history, weeks_axis, slope = gini_stability(y_val, val_preds, weeks_val)
    
    print(f"\n✅ Valid AUC:       {auc_score:.4f}")
    print(f"📉 Stability Score: {stab_score:.4f} (Slope: {slope:.4f})")
    
    if slope < 0:
        print("⚠️  WARNING: Performance is degrading over time!")
    
    return model, X_val, y_val, weeks_val, gini_history, weeks_axis

def generate_plots(model, X_val, y_val, weeks_val, gini_history, weeks_axis):
    print("🎨 Generating Plots...")
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    # PLOT 1: Stability Over Time (The most important plot for your presentation)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=weeks_axis, y=gini_history, marker='o', color='blue', label='Weekly Gini')
    
    # Add trendline
    z = np.polyfit(weeks_axis, gini_history, 1)
    p = np.poly1d(z)
    plt.plot(weeks_axis, p(weeks_axis), "r--", label=f'Trend (Slope={z[0]:.4f})')
    
    plt.title(f'Model Stability Over Time (Validation Weeks {min(weeks_axis)}-{max(weeks_axis)})')
    plt.xlabel('Week Number')
    plt.ylabel('Gini Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('stability_over_time.png')
    plt.close()
    print("Saved stability_over_time.png")

    # PLOT 2: Feature Importance
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, max_num_features=15, importance_type='split', height=0.5)
    plt.title('Top 15 Features (Robust Model)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close() 
    print("Saved feature_importance.png")

if __name__ == "__main__":
    results = run_pipeline()
    
    if results[0] is not None:
        model, X_val, y_val, weeks_val, gini_hist, weeks_ax = results
        generate_plots(model, X_val, y_val, weeks_val, gini_hist, weeks_ax)
        print("🎉 Done!")