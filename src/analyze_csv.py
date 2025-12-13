import polars as pl
import os

# --- CONFIGURATION ---
DATA_DIR = "./data/original-data/csv_files/train"  # Ensure this path matches your folder structure
OUTPUT_FILE = "static_column_analysis.txt"

def analyze_static_structure():
    print(f"🚀 Starting Static File Analysis...")
    
    # We only need to load one of the files to get the headers/schema
    # train_static_0_0 and train_static_0_1 have the exact same columns.
    file_path = f"{DATA_DIR}/train_static_0_0.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    print(f"📖 Reading schema from {file_path}...")
    # We read only the first 100 rows to get types/headers fast without loading 1GB+
    df = pl.read_csv(file_path, n_rows=100) 
    
    cols = df.columns
    total_cols = len(cols)
    
    # --- CATEGORIZE COLUMNS ---
    # Sorting them helps you see groups of related features
    cols_A = sorted([c for c in cols if c.endswith("A")]) # Amount
    cols_D = sorted([c for c in cols if c.endswith("D")]) # Date
    cols_M = sorted([c for c in cols if c.endswith("M")]) # Masked
    cols_L = sorted([c for c in cols if c.endswith("L")]) # List/String
    cols_P = sorted([c for c in cols if c.endswith("P")]) # Person
    cols_other = sorted([c for c in cols if c not in cols_A + cols_D + cols_M + cols_L + cols_P])

    # --- WRITE REPORT TO FILE ---
    print(f"📝 Writing full report to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("="*50 + "\n")
        f.write(f" HOME CREDIT - STATIC TABLE ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Columns: {total_cols}\n\n")
        
        f.write(f"--- [A] AMOUNT COLUMNS ({len(cols_A)}) ---\n")
        f.write("These are financial numbers (Income, Debt, Credit Amount).\n")
        for c in cols_A: f.write(f"  - {c}\n")
        f.write("\n")
        
        f.write(f"--- [D] DATE COLUMNS ({len(cols_D)}) ---\n")
        f.write("These are dates (Birth, Employment Start).\n")
        for c in cols_D: f.write(f"  - {c}\n")
        f.write("\n")

        f.write(f"--- [M] MASKED COLUMNS ({len(cols_M)}) ---\n")
        f.write("These are categorical features (hashed/anonymized).\n")
        for c in cols_M: f.write(f"  - {c}\n")
        f.write("\n")

        f.write(f"--- [L] LIST/STRING COLUMNS ({len(cols_L)}) ---\n")
        f.write("These are text categories (Education, Marital Status).\n")
        for c in cols_L: f.write(f"  - {c}\n")
        f.write("\n")
        
        f.write(f"--- [P] PERSON/PRODUCT COLUMNS ({len(cols_P)}) ---\n")
        for c in cols_P: f.write(f"  - {c}\n")
        f.write("\n")

        f.write(f"--- OTHER COLUMNS ({len(cols_other)}) ---\n")
        for c in cols_other: f.write(f"  - {c}\n")
        f.write("\n")

    print("✅ Done!")
    print(f"👉 Open '{OUTPUT_FILE}' to see the full list.")

if __name__ == "__main__":
    analyze_static_structure()