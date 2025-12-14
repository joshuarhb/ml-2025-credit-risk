import pandas as pd
import os
import gc
import glob
import numpy as np

def process_bureau_a_1(data_dir):
    """
    Aggregates Credit Bureau A (Depth 1) data.
    These files represent the client's credit history with OTHER institutions.
    """
    print("🏦 Engineering Credit Bureau A Features...")
    
    # 1. Find all Bureau A files (train_credit_bureau_a_1_0.csv, _1, etc.)
    # Note: In test set, it might just be test_credit_bureau_a_1_0.csv
    files = sorted(glob.glob(os.path.join(data_dir, "*credit_bureau_a_1_*.csv")))
    
    if not files:
        print("⚠️ No Credit Bureau A files found!")
        return pd.DataFrame(columns=["case_id"]).set_index("case_id")

    agg_dfs = []
    
    for f in files:
        print(f"   Processing file: {os.path.basename(f)}...")
        
        # 2. Dynamic Column Mapping
        # We want to identify 'Amount' (A), 'DPD' (P), and 'Status' (M/L)
        try:
            header = pd.read_csv(f, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError:
            continue
            
        col_map = {}
        # Columns often have messy suffixes like _12345A. We map them to generic names.
        for c in header:
            if "outstandingdebt" in c: col_map[c] = "bureau_debt"
            elif "monthlyinstlamount" in c: col_map[c] = "bureau_annuity"
            elif "overdueamount" in c: col_map[c] = "bureau_overdue"
            elif "dpd" in c and c.endswith("P"): col_map[c] = "bureau_dpd"
            elif "classificationofcontr" in c: col_map[c] = "contract_status"
            elif "purposeofcred" in c: col_map[c] = "purpose"
        
        if not col_map:
            # Fallback: Just grab ANY 'Amount' or 'DPD' column if specific names fail
            for c in header:
                if c.endswith("A") and "amount" in c: col_map[c] = f"bureau_amt_{c.split('_')[-1]}"
                if c.endswith("P"): col_map[c] = f"bureau_dpd_{c.split('_')[-1]}"

        load_cols = ["case_id"] + list(col_map.keys())
        
        # 3. Process in Chunks
        for chunk in pd.read_csv(f, usecols=load_cols, chunksize=100000, low_memory=False):
            chunk = chunk.rename(columns=col_map)
            
            # --- Feature Logic ---
            
            # Active Loans Flag
            if "contract_status" in chunk.columns:
                # Assuming '01' or similar often denotes active, but checking for 'Active' string is safer if decoded
                # In raw Kaggle data, these are often codes. We'll count unique contracts.
                pass 
            
            # Numeric Fill
            nums = [c for c in chunk.columns if "bureau_" in c]
            for c in nums:
                chunk[c] = pd.to_numeric(chunk[c], errors='coerce').fillna(0)

            # --- Aggregation ---
            # We want:
            # - Total Debt
            # - Max Overdue (Critical for risk!)
            # - Max DPD
            # - Count of active loans
            
            aggs = {
                "case_id": "count", # Total number of external loans
            }
            
            if "bureau_debt" in chunk.columns:
                aggs["bureau_debt"] = ["sum", "max"]
            if "bureau_overdue" in chunk.columns:
                aggs["bureau_overdue"] = ["sum", "max"]
            if "bureau_dpd" in chunk.columns:
                aggs["bureau_dpd"] = ["max", "mean"]
            if "bureau_annuity" in chunk.columns:
                aggs["bureau_annuity"] = "sum"
                
            agg_chunk = chunk.groupby("case_id").agg(aggs)
            
            # Flatten columns
            new_cols = []
            for c in agg_chunk.columns.values:
                if c[0] == "case_id":
                    new_cols.append("bureau_total_loans")
                else:
                    new_cols.append(f"{c[0]}_{c[1]}")
            
            agg_chunk.columns = new_cols
            agg_dfs.append(agg_chunk)
            
            del chunk
        gc.collect()

    # 4. Final Reduce
    print("   Combining Bureau chunks...")
    if not agg_dfs:
         return pd.DataFrame(columns=["case_id"]).set_index("case_id")
         
    full_df = pd.concat(agg_dfs)
    
    # We aggregate again because case_id is repeated across chunks and files
    final_aggs = {
        "bureau_total_loans": "sum",
    }
    # Add other columns dynamically
    for c in full_df.columns:
        if "max" in c: final_aggs[c] = "max"
        elif "sum" in c: final_aggs[c] = "sum"
        elif "mean" in c: final_aggs[c] = "mean"
    
    final_df = full_df.groupby("case_id").agg(final_aggs)
    
    # 5. Ratios (Interaction Features)
    # Debt to Income (Proxy) - requires joining, so we do it in main script, 
    # but here we can do Debt per Loan
    if "bureau_debt_sum" in final_df.columns and "bureau_total_loans" in final_df.columns:
        final_df["bureau_avg_debt"] = final_df["bureau_debt_sum"] / final_df["bureau_total_loans"]
        
    return final_df