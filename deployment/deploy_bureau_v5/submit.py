import pandas as pd
import joblib
import os
import glob
import gc
import numpy as np
import sys

# CONFIG - Kaggle Paths
TEST_DIR = "/kaggle/input/home-credit-credit-risk-model-stability/csv_files/test"
# Dynamic Model Directory Finding
try:
    # Looks for the dataset we uploaded via Slurm
    MODEL_DIR = [d for d in glob.glob("/kaggle/input/home-credit-model-*")][0]
except IndexError:
    MODEL_DIR = "/kaggle/input/home-credit-model-advanced-v1" # Fallback

MODEL_PATH = f"{MODEL_DIR}/model.joblib"
FEAT_PATH = f"{MODEL_DIR}/features.joblib"
CAT_PATH = f"{MODEL_DIR}/cat_cols.joblib"

# --- 1. BUREAU LOGIC (Embedded for Kaggle Safety) ---
def process_bureau_a_1(data_dir):
    """
    Aggregates Credit Bureau A (Depth 1) data.
    FIXED VERSION: Prevents Duplicate Column Names.
    """
    print("🏦 Engineering Credit Bureau A Features...")
    
    files = sorted(glob.glob(os.path.join(data_dir, "*credit_bureau_a_1_*.csv")))
    
    if not files:
        print("⚠️ No Credit Bureau A files found!")
        return pd.DataFrame(columns=["case_id"]).set_index("case_id")

    agg_dfs = []
    
    for f in files:
        print(f"   Processing file: {os.path.basename(f)}...")
        
        try:
            # Read header to create mapping
            header = pd.read_csv(f, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError:
            continue
            
        col_map = {}
        # 1. Map to unique names (preserve original suffix to ensure uniqueness)
        for c in header:
            if "outstandingdebt" in c: col_map[c] = f"bureau_debt_{c}"
            elif "monthlyinstlamount" in c: col_map[c] = f"bureau_annuity_{c}"
            elif "overdueamount" in c: col_map[c] = f"bureau_overdue_{c}"
            elif "dpd" in c and c.endswith("P"): col_map[c] = f"bureau_dpd_{c}"
        
        if not col_map:
            continue

        load_cols = ["case_id"] + list(col_map.keys())
        
        # 2. Process in Chunks
        for chunk in pd.read_csv(f, usecols=load_cols, chunksize=100000, low_memory=False):
            chunk = chunk.rename(columns=col_map)
            
            # Numeric Conversion
            bureau_cols = [c for c in chunk.columns if c.startswith("bureau_")]
            for c in bureau_cols:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce').fillna(0)

            # Define Aggregations
            aggs = {"case_id": "count"}
            for c in chunk.columns:
                if "bureau_debt" in c: aggs[c] = ["sum", "max"]
                elif "bureau_overdue" in c: aggs[c] = ["sum", "max"]
                elif "bureau_dpd" in c: aggs[c] = ["max", "mean"]
                elif "bureau_annuity" in c: aggs[c] = "sum"
            
            if "case_id" in aggs: del aggs["case_id"]
                
            agg_chunk = chunk.groupby("case_id").agg(aggs)
            
            # --- CRITICAL FIX HERE ---
            # Do NOT strip the suffix. Keep full name: bureau_debt_123A_sum
            new_cols = []
            for col_name, stat in agg_chunk.columns.values:
                new_cols.append(f"{col_name}_{stat}")
            
            agg_chunk.columns = new_cols
            # Add loan count manually
            agg_chunk["bureau_total_loans"] = chunk.groupby("case_id").size()
            
            agg_dfs.append(agg_chunk)
            del chunk
        gc.collect()

    # 3. Final Reduce
    print("   Combining Bureau chunks...")
    if not agg_dfs:
         return pd.DataFrame(columns=["case_id"]).set_index("case_id")
         
    full_df = pd.concat(agg_dfs)
    
    # 4. Dimensionality Reduction (Optional but recommended)
    # Since we have many split columns (debt_123A_sum, debt_456B_sum), 
    # we now aggregate them by case_id.
    
    # First, handle duplicates if any sneak in (Paranoia check)
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    
    # Generate final aggregation dict dynamically
    final_aggs = {}
    for c in full_df.columns:
        if "max" in c: final_aggs[c] = "max"
        elif "sum" in c: final_aggs[c] = "sum"
        elif "mean" in c: final_aggs[c] = "mean"
        elif "total_loans" in c: final_aggs[c] = "sum"
    
    final_df = full_df.groupby("case_id").agg(final_aggs)
    
    # 5. Simplify Features (Combine the disparate columns)
    # Instead of having 50 columns for debt, let's sum them into one 'total_debt'
    # This makes the model more robust and easier to interpret.
    
    print("   Simplifying Bureau Features...")
    # Find all columns related to Debt Sums
    debt_sum_cols = [c for c in final_df.columns if "bureau_debt" in c and "sum" in c]
    if debt_sum_cols:
        final_df["bureau_debt_total_sum"] = final_df[debt_sum_cols].sum(axis=1)
        # Drop the individual ones to save memory/noise
        final_df.drop(columns=debt_sum_cols, inplace=True)
        
    # Overdue Max
    overdue_max_cols = [c for c in final_df.columns if "bureau_overdue" in c and "max" in c]
    if overdue_max_cols:
        final_df["bureau_overdue_total_max"] = final_df[overdue_max_cols].max(axis=1)
        final_df.drop(columns=overdue_max_cols, inplace=True)

    # DPD Max
    dpd_max_cols = [c for c in final_df.columns if "bureau_dpd" in c and "max" in c]
    if dpd_max_cols:
        final_df["bureau_dpd_total_max"] = final_df[dpd_max_cols].max(axis=1)
        final_df.drop(columns=dpd_max_cols, inplace=True)

    return final_df

# --- 2. APPLPREV LOGIC ---
def process_applprev_advanced(data_dir):
    print("🧠 Engineering Advanced History Features...")
    files = [f for f in os.listdir(data_dir) if "applprev_1" in f and f.endswith(".csv")]
    agg_dfs = []
    
    for f in files:
        path = os.path.join(data_dir, f)
        try: header = pd.read_csv(path, nrows=0).columns.tolist()
        except: continue
        
        col_map = {}
        for c in header:
            if c.startswith("actualdpd"): col_map[c] = "dpd"
            elif c.startswith("credamount"): col_map[c] = "amount"
            elif c.startswith("status"): col_map[c] = "status"
        
        if not col_map: continue
        load_cols = ["case_id"] + list(col_map.keys())
        
        for chunk in pd.read_csv(path, usecols=load_cols, chunksize=50000):
            chunk = chunk.rename(columns=col_map)
            chunk["is_refused"] = 0
            if "status" in chunk.columns:
                chunk["is_refused"] = chunk["status"].astype(str).str.contains("D", na=False).astype(int)
            
            chunk["dpd"] = chunk["dpd"].fillna(0) if "dpd" in chunk.columns else 0
            chunk["amount"] = chunk["amount"].fillna(0) if "amount" in chunk.columns else 0

            agg = chunk.groupby("case_id").agg({
                "dpd": ["max", "mean"],
                "amount": ["max", "sum"],
                "is_refused": "sum",
                "case_id": "count"
            })
            agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
            agg.rename(columns={"case_id_count": "total_apps"}, inplace=True)
            agg_dfs.append(agg)
        gc.collect()

    if not agg_dfs: return pd.DataFrame(columns=["case_id"]).set_index("case_id")

    full_agg = pd.concat(agg_dfs)
    final_df = full_agg.groupby("case_id").agg({
        "dpd_max": "max", "dpd_mean": "mean", "amount_max": "max",
        "amount_sum": "sum", "is_refused_sum": "sum", "total_apps": "sum"
    })
    final_df["refusal_rate"] = final_df["is_refused_sum"] / final_df["total_apps"]
    final_df["avg_loan_amount"] = final_df["amount_sum"] / final_df["total_apps"]
    
    return final_df

def process_domain_features(data_dir):
    print("🧠 Engineering Domain & Trend Features...")
    
    # We focus on the 'applprev_1' files (Previous Applications)
    files = sorted(glob.glob(os.path.join(data_dir, "*applprev_1_*.csv")))
    
    if not files:
        print("⚠️ No ApplPrev files found!")
        return pd.DataFrame(columns=["case_id"]).set_index("case_id")

    agg_dfs = []
    
    for f in files:
        # Load key columns: Case ID, Status, Date, Amount, DPD
        # We need the DATE to calculate trends
        try:
            # minimal load to find names
            header = pd.read_csv(f, nrows=0).columns.tolist()
            
            # Dynamic mapping
            col_map = {}
            for c in header:
                if "creationdate" in c: col_map[c] = "date"
                elif "actualdpd" in c: col_map[c] = "dpd"
                elif "credamount" in c: col_map[c] = "amount"
                elif "annuity" in c: col_map[c] = "annuity"
                elif "status" in c: col_map[c] = "status"
                elif "cancelreason" in c: col_map[c] = "cancel_reason"
            
            load_cols = ["case_id"] + list(col_map.keys())
            
            for chunk in pd.read_csv(f, usecols=load_cols, chunksize=100000, low_memory=False):
                chunk = chunk.rename(columns=col_map)
                
                # 1. CLEANING
                chunk["date"] = pd.to_datetime(chunk["date"], errors='coerce')
                chunk["amount"] = pd.to_numeric(chunk["amount"], errors='coerce').fillna(0)
                chunk["annuity"] = pd.to_numeric(chunk["annuity"], errors='coerce').fillna(0)
                chunk["dpd"] = pd.to_numeric(chunk["dpd"], errors='coerce').fillna(0)
                
                # 2. RATIOS (Financial Health)
                # Term Length Proxy: Total Amount / Monthly Payment
                # If this is High, they have long loans. If Low, short loans.
                chunk["term_proxy"] = chunk["amount"] / (chunk["annuity"] + 1.0)
                
                # 3. RECENT VS OLD (The Trend Logic)
                # We can't do perfect time series in a chunk, BUT we can separate by status/type
                # Let's aggregate based on "Approved" vs "Refused" specifically
                
                chunk["is_approved"] = chunk["status"].astype(str).str.contains("K", na=False).astype(int)
                chunk["is_refused"] = chunk["status"].astype(str).str.contains("D", na=False).astype(int)
                
                # 4. AGGREGATION
                aggs = {
                    "case_id": "count",
                    "term_proxy": ["mean", "max"],
                    "amount": ["mean", "max"],
                    "dpd": ["mean", "max", "std"], # STD captures volatility!
                    "is_approved": "sum",
                    "is_refused": "sum"
                }
                
                agg_chunk = chunk.groupby("case_id").agg(aggs)
                
                # Flatten
                new_cols = []
                for c, stat in agg_chunk.columns:
                    if c == "case_id": new_cols.append("total_apps")
                    else: new_cols.append(f"dom_{c}_{stat}")
                
                agg_chunk.columns = new_cols
                agg_dfs.append(agg_chunk)
                del chunk
                
        except Exception as e:
            print(f"Skipping file {f} due to error: {e}")
            continue
            
        gc.collect()

    if not agg_dfs:
        return pd.DataFrame(columns=["case_id"]).set_index("case_id")

    # COMBINE CHUNKS
    print("   Consolidating Domain Features...")
    full_df = pd.concat(agg_dfs)
    
    # Final Reduce
    final_aggs = {}
    for c in full_df.columns:
        if "max" in c: final_aggs[c] = "max"
        elif "mean" in c: final_aggs[c] = "mean"
        elif "sum" in c: final_aggs[c] = "sum"
        elif "std" in c: final_aggs[c] = "max" # max volatility observed
    
    final_df = full_df.groupby("case_id").agg(final_aggs)
    
    # 5. FINAL DERIVED RATIOS (The "Deep" Feature Engineering)
    # Refusal Rate: (Refused / Total)
    # Note: 'dom_case_id_count' might not be sum of approved+refused, so we estimate
    total = final_df["dom_is_approved_sum"] + final_df["dom_is_refused_sum"] + 1
    final_df["dom_refusal_rate"] = final_df["dom_is_refused_sum"] / total
    
    # Volatility Flag: Do they have inconsistent DPD?
    if "dom_dpd_std" in final_df.columns:
        final_df["dom_volatile_income_behavior"] = (final_df["dom_dpd_std"] > 10).astype(int)
        
    return final_df

def process_financial_features(data_dir):
    print("💰 Engineering Tax & Cash Flow Features...")
    
    # --- PART A: TAX REGISTRY C (Employment/Income) ---
    # We choose 'c' because it often contains 'employername' and 'pmtamount'
    tax_files = sorted(glob.glob(os.path.join(data_dir, "*tax_registry_c_1_*.csv")))
    
    tax_dfs = []
    if tax_files:
        print(f"   Processing {len(tax_files)} Tax files...")
        for f in tax_files:
            # Columns: pmtamount_36A (Tax Amount), employername_160M, processingdate_168D
            # Note: Tax Amount is highly correlated with Salary.
            try:
                # Dynamic mapping
                header = pd.read_csv(f, nrows=0).columns.tolist()
                col_map = {}
                for c in header:
                    if "pmtamount" in c: col_map[c] = "tax_amount"
                    elif "employername" in c: col_map[c] = "employer_id"
                    elif "processingdate" in c: col_map[c] = "date"
                
                if not col_map: continue
                
                chunk = pd.read_csv(f, usecols=["case_id"] + list(col_map.keys()), low_memory=False)
                chunk = chunk.rename(columns=col_map)
                
                chunk["tax_amount"] = pd.to_numeric(chunk["tax_amount"], errors='coerce').fillna(0)
                
                # Aggregations
                aggs = {
                    "tax_amount": ["max", "mean", "sum"],
                    "employer_id": "nunique", # Job Stability: How many diff employers?
                    "case_id": "count" # How many tax records found?
                }
                
                agg_chunk = chunk.groupby("case_id").agg(aggs)
                
                # Flatten
                new_cols = []
                for c, stat in agg_chunk.columns:
                    if c == "case_id": new_cols.append("tax_record_count")
                    else: new_cols.append(f"fin_{c}_{stat}")
                
                agg_chunk.columns = new_cols
                tax_dfs.append(agg_chunk)
                del chunk
                gc.collect()
            except Exception as e:
                print(f"Error in tax: {e}")
                continue

    # --- PART B: DEBIT CARD (Liquidity) ---
    debit_files = sorted(glob.glob(os.path.join(data_dir, "*debitcard_1_*.csv")))
    
    debit_dfs = []
    if debit_files:
        print(f"   Processing {len(debit_files)} Debit files...")
        for f in debit_files:
            # Columns: last180dayaveragebalance, last30dayturnover
            try:
                chunk = pd.read_csv(f, low_memory=False)
                # Map complex names to simple ones
                col_map = {}
                for c in chunk.columns:
                    if "averagebalance" in c: col_map[c] = "balance_180d"
                    elif "turnover" in c and "30" in c: col_map[c] = "turnover_30d"
                    elif "turnover" in c and "180" in c: col_map[c] = "turnover_180d"
                
                chunk = chunk.rename(columns=col_map)
                
                # Numeric
                for c in ["balance_180d", "turnover_30d", "turnover_180d"]:
                    if c in chunk.columns:
                        chunk[c] = pd.to_numeric(chunk[c], errors='coerce').fillna(0)
                
                # Aggregations
                aggs = {}
                if "balance_180d" in chunk.columns: aggs["balance_180d"] = "mean"
                if "turnover_30d" in chunk.columns: aggs["turnover_30d"] = "mean"
                
                if not aggs: continue
                
                agg_chunk = chunk.groupby("case_id").agg(aggs)
                agg_chunk.columns = [f"fin_debit_{c}" for c in agg_chunk.columns]
                debit_dfs.append(agg_chunk)
                del chunk
                gc.collect()
            except Exception as e:
                continue

    # --- MERGE ---
    print("   Consolidating Financials...")
    final_df = pd.DataFrame(columns=["case_id"]).set_index("case_id")
    
    if tax_dfs:
        full_tax = pd.concat(tax_dfs)
        # Final reduce tax
        tax_agg = full_tax.groupby("case_id").max() # Max works for both sum/max cols
        final_df = final_df.join(tax_agg, how="outer")
        
    if debit_dfs:
        full_debit = pd.concat(debit_dfs)
        debit_agg = full_debit.groupby("case_id").mean()
        final_df = final_df.join(debit_agg, how="outer")
    
    return final_df

def process_granular_features(data_dir):
    print("🔬 Engineering Granular & Stability Features...")
    
    # --- PART A: BUREAU B (Depth 2 - Payment History) ---
    # Structure: case_id -> contract (num_group1) -> payment (num_group2)
    bureau_b_files = sorted(glob.glob(os.path.join(data_dir, "*credit_bureau_b_2_*.csv")))
    
    bureau_dfs = []
    if bureau_b_files:
        print(f"   Processing {len(bureau_b_files)} Bureau B (Payment) files...")
        for f in bureau_b_files:
            try:
                # Columns: pmts_dpdvalue_108P (Days Past Due), pmts_date_1107D (Date)
                chunk = pd.read_csv(f, low_memory=False)
                
                # Dynamic Map
                col_map = {}
                for c in chunk.columns:
                    if "pmts_dpdvalue" in c: col_map[c] = "payment_dpd"
                    elif "pmts_date" in c: col_map[c] = "payment_date"
                
                chunk = chunk.rename(columns=col_map)
                
                if "payment_dpd" not in chunk.columns: continue
                
                chunk["payment_dpd"] = pd.to_numeric(chunk["payment_dpd"], errors='coerce').fillna(0)
                
                # 1. Aggregation Level 1: Payment -> Contract
                # We want the MAX DPD for each contract (Did they ever miss a payment on this specific loan?)
                contract_agg = chunk.groupby(["case_id", "num_group1"])["payment_dpd"].max().reset_index()
                contract_agg.rename(columns={"payment_dpd": "contract_max_dpd"}, inplace=True)
                
                # 2. Aggregation Level 2: Contract -> Case
                # Now we aggregate across all contracts for the user
                case_agg = contract_agg.groupby("case_id").agg({
                    "contract_max_dpd": ["max", "mean", "sum"], # Sum captures total "badness" across all loans
                    "num_group1": "count" # Number of external loans tracked here
                })
                
                # Flatten
                case_agg.columns = [f"gran_bureau_b_{c[0]}_{c[1]}" for c in case_agg.columns]
                bureau_dfs.append(case_agg)
                
                del chunk, contract_agg
                gc.collect()
            except Exception as e:
                continue
                
    # --- PART B: PERSON (Life Stability) ---
    # Files: train_person_1.csv
    # We look for address changes (district) and role changes.
    person_files = sorted(glob.glob(os.path.join(data_dir, "*person_1_*.csv")))
    person_dfs = []
    
    if person_files:
        print(f"   Processing {len(person_files)} Person files...")
        for f in person_files:
            try:
                # We just need to count unique values for specific columns
                # This proxies for "Stability". 
                # A person with 5 different addresses in 2 years is risky.
                chunk = pd.read_csv(f, low_memory=False)
                
                aggs = {}
                # Find zip/district/role columns dynamically
                for c in chunk.columns:
                    if "zip" in c: aggs[c] = "nunique"
                    elif "district" in c: aggs[c] = "nunique"
                    elif "role" in c: aggs[c] = "nunique"
                    elif "birth" in c and c.endswith("D"): aggs[c] = "first" # Just grab birth date
                
                if not aggs: continue
                
                if "case_id" in aggs: del aggs["case_id"]
                
                agg_chunk = chunk.groupby("case_id").agg(aggs)
                
                # Rename
                new_cols = []
                for c in agg_chunk.columns:
                    if "birth" in c: new_cols.append("gran_person_birth_date")
                    else: new_cols.append(f"gran_stability_{c}_count")
                
                agg_chunk.columns = new_cols
                person_dfs.append(agg_chunk)
                del chunk
                gc.collect()
            except: continue

    # --- MERGE ---
    print("   Consolidating Granular Features...")
    final_df = pd.DataFrame(columns=["case_id"]).set_index("case_id")
    
    if bureau_dfs:
        full_bureau = pd.concat(bureau_dfs)
        # Re-aggregate in case of file splits
        bur_final = full_bureau.groupby("case_id").max()
        final_df = final_df.join(bur_final, how="outer")
        
    if person_dfs:
        full_person = pd.concat(person_dfs)
        pers_final = full_person.groupby("case_id").max()
        final_df = final_df.join(pers_final, how="outer")
        
    return final_df


# --- 3. MAIN PIPELINE ---
def run_inference():
    print("🚀 Starting Hybrid Inference V5...")
    
    # 1. Load Models & Feature Lists
    print("⏳ Loading Artifacts...")
    lgb_model = joblib.load(f"{MODEL_DIR}/model.joblib")
    cat_model = joblib.load(f"{MODEL_DIR}/cat_model.joblib")
    
    # --- FIX IS HERE ---
    # In V4, both models share the same 'features.joblib'
    lgb_feats = joblib.load(f"{MODEL_DIR}/features.joblib")
    cat_feats = lgb_feats # Same features used for both
    
    cat_cols = joblib.load(f"{MODEL_DIR}/cat_cols.joblib")

    # 2. Load Data
    df_base = pd.read_csv(f"{TEST_DIR}/test_base.csv")
    static_files = [f for f in os.listdir(TEST_DIR) if "test_static_0" in f]
    dfs = [pd.read_csv(os.path.join(TEST_DIR, f), low_memory=False) for f in static_files]
    if dfs: df_test = df_base.merge(pd.concat(dfs, ignore_index=True), on="case_id", how="left")
    else: df_test = df_base
    
    # 3. Engineer Features
    # Ensure these functions are defined above!
    try: df_hist = process_applprev_advanced(TEST_DIR)
    except: df_hist = pd.DataFrame()
    
    try: df_bur = process_bureau_a_1(TEST_DIR)
    except: df_bur = pd.DataFrame()
        
    try: df_dom = process_domain_features(TEST_DIR)
    except: df_dom = pd.DataFrame()
        
    try: df_fin = process_financial_features(TEST_DIR)
    except: df_fin = pd.DataFrame()

    try:df_gran = process_granular_features(TEST_DIR)
    except:df_gran = pd.DataFrame()
    
    for df in [df_hist, df_bur, df_dom, df_fin, df_gran]:
        if not df.empty: df_test = df_test.merge(df, on="case_id", how="left")
    
    # 4. PREDICT LIGHTGBM
    # Fill missing cols with 0
    for c in lgb_feats:
        if c not in df_test.columns: df_test[c] = 0
    
    X_lgb = df_test[lgb_feats].copy()
    for c in cat_cols: 
        if c in X_lgb.columns: X_lgb[c] = X_lgb[c].astype('category')
            
    p_lgb = lgb_model.predict_proba(X_lgb)[:, 1]
    
    # 5. PREDICT CATBOOST
    # (Same features, but different type handling)
    X_cat = df_test[cat_feats].copy()
    for c in cat_cols:
        if c in X_cat.columns: X_cat[c] = X_cat[c].astype(str).fillna("Missing")
            
    p_cat = cat_model.predict_proba(X_cat)[:, 1]
    
    # 6. BLEND & SUBMIT
    final_score = (p_lgb + p_cat) / 2.0
    
    pd.DataFrame({
        "case_id": df_test["case_id"],
        "score": final_score
    }).to_csv("submission.csv", index=False)
    print("✅ Submission Saved")

if __name__ == "__main__":
    run_inference()