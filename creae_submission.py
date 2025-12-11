import polars as pl
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
TEST_DIR = "./data/original-data/csv_files/test" 
MODEL_PATH = "lgbm_baseline.joblib"
CAT_LIST_PATH = "cat_cols.joblib"

def run_submission():
    print("🚀 Starting Submission Pipeline...")
    
    # 1. Load Model and Category List
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found!")
        return
    if not os.path.exists(CAT_LIST_PATH):
        print("❌ Error: Category list not found! You must re-run baseline_model.py")
        return
    
    model = joblib.load(MODEL_PATH)
    train_cat_cols = joblib.load(CAT_LIST_PATH)
    print("✅ Model and Metadata Loaded.")

    # 2. Load Test Data
    try:
        df_base = pl.read_csv(f"{TEST_DIR}/test_base.csv")
        df_static_0 = pl.read_csv(f"{TEST_DIR}/test_static_0_0.csv")
        df_static_1 = pl.read_csv(f"{TEST_DIR}/test_static_0_1.csv")
        
        df_static = pl.concat([df_static_0, df_static_1], how="vertical_relaxed")
        df_test = df_base.join(df_static, on="case_id", how="left")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    # 3. Preprocessing
    print("🧹 Preprocessing Test Data...")
    pdf_test = df_test.to_pandas()
    
    # Build X_test using dictionary
    expected_features = model.feature_name_
    data_dict = {}
    
    print("⚙️ Aligning features...")
    for feature in expected_features:
        if feature in pdf_test.columns:
            data_dict[feature] = pdf_test[feature]
        else:
            # Use numpy nan or None instead of pd.NA to be safer with floats
            data_dict[feature] = None 
            
    X_test = pd.DataFrame(data_dict)
    
    # --- STEP 4a: Force Known Categoricals ---
    print("🔧 restoring Categorical Types...")
    for col in train_cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')

    # --- STEP 4b: THE NEW FIX (Force Numerics) ---
    # Any column that is NOT a category but is still 'object' must be forced to float.
    # This catches columns like 'deferredmnthsnum_166L' that were numeric in train
    # but became objects in test due to missing data.
    
    print("🧹 Cleaning remaining object columns...")
    # Find columns that are still 'object'
    obj_cols = X_test.select_dtypes(include=['object']).columns
    
    for col in obj_cols:
        # force to numeric, turn errors (strings) into NaN
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # 5. Predict
    print("🔮 Generating Predictions...")
    scores = model.predict_proba(X_test)[:, 1]

    # 6. Save
    submission = pd.DataFrame({
        "case_id": df_test["case_id"],
        "score": scores
    })
    
    submission.to_csv("submission.csv", index=False)
    print("✅ submission.csv saved successfully!")
    print(submission.head())

if __name__ == "__main__":
    run_submission()