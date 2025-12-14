import pandas as pd
import os
import glob
import numpy as np

# CONFIG
DATA_DIR = "./data/raw/csv_files/train"
OUTPUT_FILE = "reports/dataset_stats.txt"

def scan_dataset():
    print("📊 Starting Dataset Scan...")
    
    # ensure reports dir exists
    os.makedirs("reports", exist_ok=True)
    
    stats = []
    total_features = 0
    total_rows_history = 0
    
    # Get all CSV files
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("=== HOME CREDIT DATASET STATISTICS ===\n\n")
        
        for path in files:
            filename = os.path.basename(path)
            print(f"   Scanning {filename}...")
            
            # 1. Get Columns (Fast)
            # Read only header
            header = pd.read_csv(path, nrows=0).columns.tolist()
            num_cols = len(header)
            total_features += num_cols
            
            # 2. Get Row Count (Memory Safe)
            # We count lines in the file directly (fastest way in Linux)
            # Subtract 1 for header
            try:
                with open(path) as myfile:
                    num_rows = sum(1 for line in myfile) - 1
            except:
                num_rows = 0 # Fallback
                
            # Track massive numbers for depth tables
            if "applprev" in filename or "bureau" in filename:
                total_rows_history += num_rows

            # Write details to report
            f.write(f"📂 File: {filename}\n")
            f.write(f"   - Dimensions: {num_rows:,} rows x {num_cols} columns\n")
            f.write(f"   - Columns: {', '.join(header[:5])} ...\n") # Show first 5 cols
            f.write("-" * 30 + "\n")
            
            stats.append({
                "File": filename,
                "Rows": num_rows,
                "Cols": num_cols
            })
        
        # 3. CALCULATE "BIG NUMBERS"
        # Base table tells us strictly how many unique case_ids (clients) there are
        base_file = [s for s in stats if "train_base" in s["File"]]
        total_clients = base_file[0]["Rows"] if base_file else 0
        
        f.write("\n=== 📢 THE 'BIG NUMBERS' FOR SLIDES ===\n")
        f.write(f"1. Total Clients (Training):  {total_clients:,}\n")
        f.write(f"2. Total Raw Files:           {len(files)}\n")
        f.write(f"3. Total Raw Features:        {total_features:,}\n")
        f.write(f"4. Total History Records:     {total_rows_history:,} (Depth-1 interactions processed)\n")
        f.write(f"5. Avg History per Client:    {total_rows_history / total_clients:.1f} records\n")
        
    print(f"✅ Scan Complete. Stats saved to {OUTPUT_FILE}")
    print(f"📄 Read it with: cat {OUTPUT_FILE}")

if __name__ == "__main__":
    scan_dataset()