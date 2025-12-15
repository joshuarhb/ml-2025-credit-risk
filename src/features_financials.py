import pandas as pd
import os
import glob
import gc

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