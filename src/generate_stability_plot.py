import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np

# 1. Load your Model and Data
MODEL_DIR = "./models/enhanced_advanced_v1" # ADJUST TO YOUR LATEST MODEL PATH
DATA_DIR = "./data/raw/csv_files/train"

print("⏳ Loading Model...")
model = joblib.load(f"{MODEL_DIR}/model.joblib")
features = joblib.load(f"{MODEL_DIR}/features.joblib")
cat_cols = joblib.load(f"{MODEL_DIR}/cat_cols.joblib")

print("⏳ Loading Validation Data (re-creating the split)...")
# We re-load just enough to reproduce the validation set
df_base = pd.read_csv(f"{DATA_DIR}/train_base.csv")
static_files = [f"./data/raw/csv_files/train/{f}" for f in ["train_static_0_0.csv", "train_static_0_1.csv"]]
df_static = pd.concat([pd.read_csv(f) for f in static_files], ignore_index=True)

# Join Base + Static (We can skip applprev for this visualization to save RAM if needed, 
# but ideally you load the full features used in training)
df = df_base.merge(df_static, on="case_id", how="left")

# IMPORTANT: If your model expects "dpd_max" etc, you MUST load them. 
# If you processed them to a CSV/Parquet, load that. 
# If not, fill missing columns with 0 just to get the plot generated quickly.
missing_cols = set(features) - set(df.columns)
for c in missing_cols:
    df[c] = 0 # Dummy fill for visualization purposes if you don't have the processed file saved

# Time Split
df["date_decision"] = pd.to_datetime(df["date_decision"])
df = df.sort_values("date_decision")
split_idx = int(len(df) * 0.9)
val_df = df.iloc[split_idx:].copy()

# 2. Predict
print("🔮 Predicting on Validation Set...")
# Ensure categories
for c in cat_cols:
    if c in val_df.columns:
        val_df[c] = val_df[c].astype('category')

X_val = val_df[features]
y_val = val_df["target"]
val_df["score"] = model.predict_proba(X_val)[:, 1]

# 3. Calculate Weekly Gini
print("📊 Calculating Stability...")
def gini_score(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1

weekly_scores = val_df.groupby("WEEK_NUM").apply(
    lambda x: gini_score(x["target"], x["score"])
).reset_index(name="gini")

# 4. Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_scores, x="WEEK_NUM", y="gini", marker="o", linewidth=2, color="#2ecc71")

# Add Trendline
z = np.polyfit(weekly_scores["WEEK_NUM"], weekly_scores["gini"], 1)
p = np.poly1d(z)
plt.plot(weekly_scores["WEEK_NUM"], p(weekly_scores["WEEK_NUM"]), "r--", alpha=0.8, label="Trend (Stability)")

plt.title("Model Stability Over Time (Validation Set)", fontsize=16)
plt.ylabel("Gini Score", fontsize=12)
plt.xlabel("Week Number", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/final_stability_plot.png")
print("✅ Plot saved to reports/figures/final_stability_plot.png")