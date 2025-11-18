import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
# FORCE HEADLESS BACKEND (Crucial for HPC)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os

# --- CONFIGURATION ---
DATA_DIR = "./data/original-data/csv_files/train"  # Ensure this path matches your folder structure
random_state = 42

def run_pipeline():
    print("⏳ Loading Data with Polars...")
    
    # 1. Load Data
    try:
        df_base = pl.read_csv(f"{DATA_DIR}/train_base.csv")
        df_static_0 = pl.read_csv(f"{DATA_DIR}/train_static_0_0.csv")
        df_static_1 = pl.read_csv(f"{DATA_DIR}/train_static_0_1.csv")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in data dir: {os.listdir(DATA_DIR)}")
        return None, None, None

    # 2. Join
    df_static = pl.concat([df_static_0, df_static_1], how="vertical_relaxed")
    print("🔗 Joining Tables...")
    df_train = df_base.join(df_static, on="case_id", how="left")
    
    # Cleanup
    del df_static_0, df_static_1, df_static
    gc.collect()
    
    # 3. Preprocessing
    print("🧹 Preprocessing...")
    null_counts = df_train.null_count()
    total_rows = df_train.height
    cols_to_keep = [col for col in df_train.columns if null_counts[col][0] / total_rows < 0.5]
    df_train = df_train.select(cols_to_keep)
    
    # Convert to Pandas
    pdf_train = df_train.to_pandas()
    
    # Handle Categoricals
    cat_cols = pdf_train.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        pdf_train[col] = pdf_train[col].astype('category')
        
    # 4. Split
    X = pdf_train.drop(columns=['target', 'case_id', 'date_decision', 'WEEK_NUM'])
    y = pdf_train['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    # 5. Train
    print("🚀 Training Baseline Model...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=random_state,
        n_jobs=-1 
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # 6. Score
    val_preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_preds)
    print(f"\n✅ Baseline AUC Score: {auc_score:.4f}")
    
    return model, pdf_train

# --- EXECUTION ---
if __name__ == "__main__":
    model, df_full = run_pipeline()
    
    if model is not None:
        print("🎨 Generating Plots...")
        
        # PLOT 1: Target Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x='target', data=df_full)
        plt.title('Target Distribution (0 = Repaid, 1 = Default)')
        plt.yscale('log')
        plt.savefig('target_distribution.png')
        print("Saved target_distribution.png")
        plt.close()

        # PLOT 2: Feature Importance
        plt.figure(figsize=(10, 8))
        lgb.plot_importance(model, max_num_features=15, importance_type='split', height=0.5)
        plt.title('Top 15 Features (Baseline Model)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Saved feature_importance.png")
        plt.close()
        
        print("🎉 Done! Check your folder for the .png files.")