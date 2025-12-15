import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score

# Config
MODEL_DIR = "./models/enhanced_bureau_v5"
DATA_DIR = "./data/raw/csv_files/train"

print("⏳ Loading Validation Data for Stability Plot...")
# We need Base + Predictions. We can mock this by loading just enough to predict.
# NOTE: For a perfect plot, you need the full engineered X_val. 
# If you don't have X_val saved, we can plot the TRAIN vs VALID performance over time
# using the 'date_decision' column.

df_base = pd.read_csv(f"{DATA_DIR}/train_base.csv")
df_base["date_decision"] = pd.to_datetime(df_base["date_decision"])

# Load predictions if you saved them, OR simple logic:
# Let's visualize the TARGET distribution stability vs WEEK to show the challenge
# (Since regenerating X_val takes 20 mins, let's plot the Data Drift first)

stability_df = df_base.groupby("WEEK_NUM")["target"].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=stability_df, x="WEEK_NUM", y="target", color="#e74c3c", label="Default Rate")

# Add trendline
z = np.polyfit(stability_df["WEEK_NUM"], stability_df["target"], 1)
p = np.poly1d(z)
plt.plot(stability_df["WEEK_NUM"], p(stability_df["WEEK_NUM"]), "k--", alpha=0.5, label="Drift Trend")

plt.title("The Stability Challenge: Default Rate Volatility over Time", fontsize=15)
plt.xlabel("Week Number")
plt.ylabel("Default Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/data_stability_challenge.png")
print("✅ Saved Data Stability Plot")