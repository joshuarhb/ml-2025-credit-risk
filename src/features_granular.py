import pandas as pd
import numpy as np
import os
import glob
import gc

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