import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.use('Agg') # Force headless backend for mplib (needed for HPC)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc
import joblib
import os

# --- CONFIGURATION ---
DATA_DIR = "./data/original-data/csv_files/train"  # Ensure this path matches your folder structure
MODEL_PATH = "lgbm_baseline.joblib"
random_state = 42

def run_pipeline():
    print("⏳ Loading Data with Polars...")
    
    # --- STEP 1: ALWAYS LOAD DATA ---
    # You still need the data to create X_val and y_val for plotting!
    try:
        df_base = pl.read_csv(f"{DATA_DIR}/train_base.csv")
        df_static_0 = pl.read_csv(f"{DATA_DIR}/train_static_0_0.csv")
        df_static_1 = pl.read_csv(f"{DATA_DIR}/train_static_0_1.csv")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None

    # Join and Process
    df_static = pl.concat([df_static_0, df_static_1], how="vertical_relaxed")
    print("🔗 Joining Tables...")
    df_train = df_base.join(df_static, on="case_id", how="left")
    del df_static_0, df_static_1, df_static
    gc.collect()
    
    print("🧹 Preprocessing...")
    null_counts = df_train.null_count()
    total_rows = df_train.height
    cols_to_keep = [col for col in df_train.columns if null_counts[col][0] / total_rows < 0.6]
    df_train = df_train.select(cols_to_keep)
    
    pdf_train = df_train.to_pandas()
    
    cat_cols = pdf_train.select_dtypes(include=['object', 'string']).columns
    for col in cat_cols:
        pdf_train[col] = pdf_train[col].astype('category')
        
    X = pdf_train.drop(columns=['target', 'case_id', 'date_decision', 'WEEK_NUM'])
    y = pdf_train['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    # --- STEP 2: CHECK FOR SAVED MODEL ---
    
    if os.path.exists(MODEL_PATH):
        print(f"💾 Found saved model at '{MODEL_PATH}'. Loading...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"⚠️ No saved model found. Training new model...")
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
        
        # SAVE THE MODEL IMMEDIATELY AFTER TRAINING
        print(f"💾 Saving model to '{MODEL_PATH}'...")
        joblib.dump(model, MODEL_PATH)
        
        print(f"💾 Saving category list to 'cat_cols.joblib'...")
        # cat_cols was defined earlier in the script
        joblib.dump(cat_cols, "cat_cols.joblib")
    
    # --- STEP 3: SCORE & RETURN ---
    # Even if we loaded from disk, we want to see the score on the current validation set
    val_preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_preds)
    print(f"\n✅ Baseline AUC Score: {auc_score:.4f}")
    
    return model, X_val, y_val, pdf_train

def generate_extra_plots(model, X_val, y_val):
    print("🎨 Generating Advanced Plots...")
    
    # Get predictions
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred_binary = (y_pred_prob > 0.05).astype(int) # Threshold 0.5

    # --- PLOT 1: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.close()
    print("Saved roc_curve.png")

    # --- PLOT 2: Confusion Matrix ---
    cm = confusion_matrix(y_val, y_pred_binary)
    # Normalize it to percentages so it's easier to read
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                xticklabels=['Pred: Repay', 'Pred: Default'],
                yticklabels=['Actual: Repay', 'Actual: Default'])
    plt.title('Confusion Matrix (Normalized)')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved confusion_matrix.png")

    # --- PLOT 3: Top 10 Missing Columns (Nullity) ---
    # We need the dataframe for this. Assuming you have 'X_val' as a DataFrame
    # Calculate missing %
    missing_series = X_val.isnull().sum() / len(X_val) * 100
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_series.values, y=missing_series.index, palette='viridis')
    plt.title('Top 10 Features with Highest Missing Values (%)')
    plt.xlabel('Percentage Missing')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('missing_values.png')
    plt.close()
    print("Saved missing_values.png")

# --- EXECUTION ---
if __name__ == "__main__":
    # Unpack all 4 variables here
    model, X_val, y_val, df_full = run_pipeline()
    
    if model is not None:
        # Now X_val and y_val exist in this scope!
        generate_extra_plots(model, X_val, y_val)

        print("🎨 Generating Basic Plots...")
        
        # PLOT 4: Target Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x='target', data=df_full)
        plt.title('Target Distribution (0 = Repaid, 1 = Default)')
        plt.yscale('log')
        plt.savefig('target_distribution.png')
        print("Saved target_distribution.png")
        plt.close()

        # PLOT 5: Feature Importance
        plt.figure(figsize=(10, 8))
        lgb.plot_importance(model, max_num_features=15, importance_type='split', height=0.5)
        plt.title('Top 15 Features (Baseline Model)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Saved feature_importance.png")
        plt.close()        
        
        print("🎉 Done! Check your folder for the .png files.")