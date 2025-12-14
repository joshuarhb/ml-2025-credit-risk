import pandas as pd
import os
import gc
import numpy as np

def process_applprev_advanced(data_dir):
    """
    Advanced Feature Engineering using Streaming (Chunking).
    Calculates Ratios and Status rates instead of just raw maximums.
    """
    print("🧠 Engineering Advanced History Features...")
    
    files = [f for f in os.listdir(data_dir) if "applprev_1" in f and f.endswith(".csv")]
    agg_dfs = []
    
    for f in files:
        path = os.path.join(data_dir, f)
        print(f"   Mining {f}...")
        
        # 1. Dynamic Column Detection
        header = pd.read_csv(path, nrows=0).columns.tolist()
        
        # Map varying column names to standard internal names
        # We use a dictionary to be flexible
        col_map = {}
        
        # Find critical columns based on prefixes
        for c in header:
            if c.startswith("actualdpd"): col_map[c] = "dpd"           # Days Past Due
            elif c.startswith("credamount"): col_map[c] = "amount"     # Loan Amount
            elif c.startswith("status"): col_map[c] = "status"         # Approved/Refused
            elif c.startswith("creationdate"): col_map[c] = "date"     # Date of application
        
        if not col_map:
            print(f"      ⚠️ Skipping {f} (No useful columns found)")
            continue
            
        # Add case_id to the load list
        load_cols = ["case_id"] + list(col_map.keys())
        
        # 2. Process in Chunks
        for chunk in pd.read_csv(path, usecols=load_cols, chunksize=100000):
            # Rename for consistency
            chunk = chunk.rename(columns=col_map)
            
            # --- FEATURE LOGIC ---
            
            # A. Refused Counter (If status exists)
            if "status" in chunk.columns:
                # Assuming 'D' or 'Refused' indicates rejection (simplified check)
                # In this data, 'A'=Approved, 'D'=Denied/Refused typically
                # We'll just count uniques to capture "complexity" of history
                chunk["is_refused"] = chunk["status"].astype(str).str.contains("D", na=False).astype(int)
            else:
                chunk["is_refused"] = 0
            
            # B. Zero out NaNs for math
            if "dpd" in chunk.columns:
                chunk["dpd"] = chunk["dpd"].fillna(0)
            else:
                chunk["dpd"] = 0
                
            if "amount" in chunk.columns:
                chunk["amount"] = chunk["amount"].fillna(0)
            else:
                chunk["amount"] = 0

            # --- AGGREGATION ---
            # We aggregate per chunk, then re-aggregate the results
            agg = chunk.groupby("case_id").agg({
                "dpd": ["max", "mean"],
                "amount": ["max", "sum"],
                "is_refused": "sum",
                "case_id": "count" # Total applications
            })
            
            # Flatten Header
            agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
            agg.rename(columns={"case_id_count": "total_apps"}, inplace=True)
            
            # C. "Interaction" Feature (Risk intensity)
            # DPD * Amount = "Value at Risk"
            # We can't do row-wise interaction here easily without exploding memory,
            # so we rely on the aggregated stats later.
            
            agg_dfs.append(agg)
            del chunk
        
        gc.collect()

    if not agg_dfs:
        return pd.DataFrame()

    print("   Consolidating History...")
    full_agg = pd.concat(agg_dfs)
    
    # Final Reduction
    final_df = full_agg.groupby("case_id").agg({
        "dpd_max": "max",
        "dpd_mean": "mean",
        "amount_max": "max",
        "amount_sum": "sum",
        "is_refused_sum": "sum",
        "total_apps": "sum"
    })
    
    # --- POST-AGGREGATION RATIOS (The "Improvement") ---
    # 1. Refusal Rate: What % of their apps get denied?
    final_df["refusal_rate"] = final_df["is_refused_sum"] / final_df["total_apps"]
    
    # 2. Average Loan Size
    final_df["avg_loan_amount"] = final_df["amount_sum"] / final_df["total_apps"]
    
    # 3. Risk Flag (High DPD)
    final_df["has_severe_default"] = (final_df["dpd_max"] > 90).astype(int)

    print(f"✅ Engineered 9 Advanced Features for {len(final_df)} cases.")
    return final_df