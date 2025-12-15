import pandas as pd
import numpy as np
import os
import glob
import gc

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