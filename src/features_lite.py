import pandas as pd
import os
import gc

def process_applprev_lite(data_dir):
    """
    Reads CSV in chunks to calculate history features without blowing up RAM.
    """
    print("🔨 Processing Previous Applications (Lite)...")
    
    # Define the files we want to mine for history
    files = [f for f in os.listdir(data_dir) if "applprev_1" in f and f.endswith(".csv")]
    
    # We only want these specific columns (Behavioral risk indicators)
    # actualdpd: Days Past Due (How late were they?)
    # creationdate: When did they apply?
    use_cols = ["case_id", "actualdpd_943P", "credamount_770A", "creationdate_885D"]
    
    agg_dfs = []
    
    for f in files:
        path = os.path.join(data_dir, f)
        print(f"   Scanning {f}...")
        
        # Read in chunks of 100k rows
        for chunk in pd.read_csv(path, usecols=lambda c: c in use_cols, chunksize=100000):
            # Standardize column names (remove the random suffix like _943P)
            chunk.columns = [c.split("_")[0] if "_" in c and c != "case_id" else c for c in chunk.columns]
            
            # Simple Aggregations on the chunk
            # Note: This is an approximation for 'max', but exact enough for 'count'
            agg = chunk.groupby("case_id").agg({
                "actualdpd": "max",
                "credamount": ["max", "mean"],
                "case_id": "count" # Counts number of previous apps
            })
            
            # Flatten multi-index columns
            agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
            agg.rename(columns={"case_id_count": "prev_app_count"}, inplace=True)
            
            agg_dfs.append(agg)
            del chunk
        gc.collect()

    print("   Combining chunks...")
    # Combine all chunk aggregations
    full_agg = pd.concat(agg_dfs)
    
    # Final Groupby to merge the chunks for the same case_id
    final_df = full_agg.groupby("case_id").agg({
        "actualdpd_max": "max",
        "credamount_max": "max",
        "credamount_mean": "mean",
        "prev_app_count": "sum"
    })
    
    print(f"✅ History Features engineered for {len(final_df)} cases.")
    return final_df