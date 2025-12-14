import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import pandas as pd

MODEL_DIR = "./models/enhanced_advanced_v1" # Adjust tag
model = joblib.load(f"{MODEL_DIR}/model.joblib")
features = joblib.load(f"{MODEL_DIR}/features.joblib")

# Get Importance
importance = model.feature_importances_
feat_imp = pd.DataFrame({"feature": features, "importance": importance})
feat_imp = feat_imp.sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(feat_imp["feature"], feat_imp["importance"], color="#3498db")
plt.gca().invert_yaxis()
plt.title("Top 20 Features Driving Credit Risk")
plt.xlabel("Split Importance")
plt.tight_layout()
plt.savefig("reports/figures/final_feature_importance.png")