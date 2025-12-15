import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Load your best model
MODEL_DIR = "./models/enhanced_bureau_v5" # Adjust if needed
model = joblib.load(f"{MODEL_DIR}/model.joblib")
features = joblib.load(f"{MODEL_DIR}/features.joblib")

# Get Importance
imp_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

# Categorize Features based on your prefixes
def get_source(name):
    if name.startswith("fin_"): return "Financial (Tax/Debit)"
    if name.startswith("gran_"): return "Granular (Depth-2)"
    if name.startswith("dom_"): return "Domain (Trends)"
    if name.startswith("bureau_"): return "Bureau (External)"
    if name.startswith("applprev_"): return "History (Internal)"
    return "Static (Baseline)"

imp_df["Source"] = imp_df["feature"].apply(get_source)

# Sort and Take Top 20
top_df = imp_df.sort_values("importance", ascending=False).head(20)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(data=top_df, y="feature", x="importance", hue="Source", dodge=False)
plt.title("Top 20 Features: The Impact of Domain Engineering", fontsize=15)
plt.xlabel("LightGBM Split Importance")
plt.ylabel("")
plt.legend(title="Data Source", loc="lower right")
plt.tight_layout()
plt.savefig("reports/figures/final_importance_colored.png")
print("✅ Saved Source-Aware Importance Plot")