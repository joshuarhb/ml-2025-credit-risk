import polars as pl
import lightgbm as lgb
import os
import glob
import numpy as np
import gc
import joblib
from sklearn.metrics import roc_auc_score
from scipy.stats import linregress

# --- CONFIGURATION ---
# Paths match your tree structure
BASE_DIR = "./data/raw/parquet_files/train"
AGGR_DIR = "./data/processed/aggregated_train"
MODEL_PATH = "./models/robust_v1/lgbm_parquet.joblib"

# Ensure directories exist
os.makedirs(AGGR_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# --- FILE GROUPS (From Notebook) ---
BASE_F = "base.parquet"
DEPTH0_F = [
    ["static_0_0.parquet", "static_0_1.parquet"],
    ["static_cb_0.parquet"]
]
DEPTH1_F = [
    ["applprev_1_0.parquet", "applprev_1_1.parquet"],
    ["other_1.parquet"],
    ["deposit_1.parquet"],
    ["person_1.parquet"],
    ["debitcard_1.parquet"],
    ["tax_registry_a_1.parquet"],
    ["tax_registry_b_1.parquet"],
    ["tax_registry_c_1.parquet"],
    ["credit_bureau_a_1_0.parquet", "credit_bureau_a_1_1.parquet", "credit_bureau_a_1_2.parquet", "credit_bureau_a_1_3.parquet"],
    ["credit_bureau_b_1.parquet"],
    ["credit_bureau_b_2.parquet"],
    ["credit_bureau_a_2_0.parquet", "credit_bureau_a_2_1.parquet", "credit_bureau_a_2_2.parquet", "credit_bureau_a_2_3.parquet",
     "credit_bureau_a_2_4.parquet", "credit_bureau_a_2_5.parquet", "credit_bureau_a_2_6.parquet", "credit_bureau_a_2_7.parquet",
     "credit_bureau_a_2_8.parquet", "credit_bureau_a_2_9.parquet", "credit_bureau_a_2_10.parquet"],
    ["applprev_2.parquet"],
    ["person_2.parquet"]
]

# --- HELPER FUNCTIONS (From Notebook) ---

def set_table_dtypes(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf_schema = lf.collect_schema()
    cast_exprs = []
    for col, dtype in lf_schema.items():
        if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
            cast_exprs.append(pl.col(col).cast(pl.Int64))
        elif col in ["date_decision"]:
            cast_exprs.append(pl.col(col).cast(pl.Date))
        elif col[-1] in ("P", "A"):
            cast_exprs.append(pl.col(col).cast(pl.Float64))
        elif col[-1] in ("M",):
            cast_exprs.append(pl.col(col).cast(pl.String))
        elif col[-1] in ("D",):
            cast_exprs.append(pl.col(col).cast(pl.Date))
        elif col[-1] in ("T", "L") and dtype == pl.Null:
            cast_exprs.append(pl.col(col).cast(pl.Float64))
    return lf.with_columns(cast_exprs)

def reduce_mem_usage(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    names = schema.names()
    dtypes = schema.dtypes()
    
    int_cols = [c for c, dt in zip(names, dtypes) if dt in (pl.Int64, pl.Int32, pl.Int16, pl.Int8)]
    float_cols = [c for c, dt in zip(names, dtypes) if dt in (pl.Float64, pl.Float32)]

    if not int_cols and not float_cols:
        return lf

    stats_exprs = []
    for c in int_cols + float_cols:
        stats_exprs.append(pl.col(c).min().alias(f"{c}__min"))
        stats_exprs.append(pl.col(c).max().alias(f"{c}__max"))

    stats = lf.select(stats_exprs).collect()
    cast_exprs = []

    for c in int_cols:
        c_min = stats[0, f"{c}__min"]
        c_max = stats[0, f"{c}__max"]
        if c_min is None or c_max is None: continue
        
        if np.iinfo(np.int8).min <= c_min and c_max <= np.iinfo(np.int8).max:
            cast_exprs.append(pl.col(c).cast(pl.Int8))
        elif np.iinfo(np.int16).min <= c_min and c_max <= np.iinfo(np.int16).max:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        elif np.iinfo(np.int32).min <= c_min and c_max <= np.iinfo(np.int32).max:
            cast_exprs.append(pl.col(c).cast(pl.Int32))

    for c in float_cols:
        if schema[c] != pl.Float64: continue
        c_min = stats[0, f"{c}__min"]
        c_max = stats[0, f"{c}__max"]
        if c_min is None or c_max is None: continue

        if np.isfinite(c_min) and np.isfinite(c_max):
            if np.finfo(np.float32).min <= c_min and c_max <= np.finfo(np.float32).max:
                cast_exprs.append(pl.col(c).cast(pl.Float32))

    return lf.with_columns(cast_exprs) if cast_exprs else lf

def handle_dates(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    cols = schema.names()
    if "date_decision" not in cols:
        to_drop = [c for c in ("date_decision", "MONTH") if c in cols]
        return lf.drop(to_drop) if to_drop else lf

    d_cols = [c for c in cols if c.endswith("D") and c != "date_decision"]
    if d_cols:
        lf = lf.with_columns([(pl.col(c) - pl.col("date_decision")).dt.total_days().alias(c) for c in d_cols])

    to_drop = [c for c in ("date_decision", "MONTH") if c in lf.collect_schema().names()]
    return lf.drop(to_drop) if to_drop else lf

def filter_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    protected = {"target", "case_id", "WEEK_NUM", "date_decision"}
    schema = lf.collect_schema()
    drop_cols = [name for name, dtype in zip(schema.names(), schema.dtypes()) if dtype == pl.Null and name not in protected]
    
    str_cols = [name for name, dtype in zip(schema.names(), schema.dtypes()) if dtype == pl.Utf8 and name not in protected]
    if str_cols:
        nunique_df = lf.select([pl.col(c).n_unique().alias(c) for c in str_cols]).collect()
        for c in str_cols:
            if nunique_df[0, c] == 1 or nunique_df[0, c] > 200:
                drop_cols.append(c)
    
    return lf.drop(drop_cols)

def build_agg_exprs(tbl: pl.LazyFrame) -> list:
    cols = tbl.collect_schema().names()
    cols_A = [c for c in cols if c.endswith("A")]
    cols_P = [c for c in cols if c.endswith("P")]
    cols_D = [c for c in cols if c.endswith("D")]
    cols_M = [c for c in cols if c.endswith("M")]
    cols_TL = [c for c in cols if c.endswith(("T", "L"))]

    agg_exprs = []
    agg_exprs += [pl.col(c).sum().alias(f"{c}_sum") for c in cols_A]
    agg_exprs += [pl.col(c).mean().alias(f"{c}_mean") for c in cols_A]
    agg_exprs += [pl.col(c).max().alias(f"{c}_max") for c in cols_A]
    agg_exprs += [pl.col(c).max().alias(f"{c}_max") for c in cols_P]
    agg_exprs += [pl.col(c).mean().alias(f"{c}_mean") for c in cols_P]
    agg_exprs += [pl.col(c).min().alias(f"{c}_min") for c in cols_D]
    agg_exprs += [pl.col(c).max().alias(f"{c}_max") for c in cols_D]
    agg_exprs += [pl.col(c).n_unique().alias(f"{c}_nunique") for c in cols_M]
    agg_exprs += [pl.col(c).max().alias(f"{c}_max") for c in cols_M]
    agg_exprs += [pl.col(c).mean().alias(f"{c}_mean") for c in cols_TL]
    agg_exprs += [pl.col(c).max().alias(f"{c}_max") for c in cols_TL]
    return agg_exprs

def process_group(group: list, depth=0, prefix: str="train"):
    raw_paths = [os.path.join(BASE_DIR, f"{prefix}_{p}") for p in group]
    batches = []
    for p in raw_paths:
        lf = pl.scan_parquet(p)
        lf = set_table_dtypes(lf)
        if depth != 0:
            agg_exprs = build_agg_exprs(lf)
            lf = lf.group_by("case_id").agg(agg_exprs)
        batches.append(lf)

    tbl = pl.concat(batches, how="vertical_relaxed")
    if len(batches) > 1:
        tbl = tbl.unique(subset=["case_id"])
    
    tbl = filter_cols(tbl)
    tbl = reduce_mem_usage(tbl)
    tbl = handle_dates(tbl)

    filename = os.path.splitext(group[0])[0]
    agg_path = os.path.join(AGGR_DIR, f"aggr_{filename}.parquet")
    tbl.collect().write_parquet(agg_path)
    print(f"✅ Aggregated {group} -> {agg_path}")
    return agg_path

def gini_stability(y_true, y_pred, weeks, w_fallingrate=88.0, w_resstd=-0.5):
    """Calculates the stability metric used in the competition."""
    gini_in_time = []
    possible_weeks = sorted(np.unique(weeks))
    
    for w in possible_weeks:
        mask = (weeks == w)
        if mask.sum() < 10: continue
        auc_w = roc_auc_score(y_true[mask], y_pred[mask])
        gini_in_time.append(2 * auc_w - 1)
    
    gini_in_time = np.array(gini_in_time)
    x = np.arange(len(gini_in_time))
    slope, intercept, _, _, _ = linregress(x, gini_in_time)
    
    falling_penalty = min(0, slope) * w_fallingrate
    std_penalty = np.std(gini_in_time - (intercept + slope * x)) * w_resstd
    stability_score = np.mean(gini_in_time) + falling_penalty + std_penalty
    
    return stability_score, slope, gini_in_time

# --- MAIN PIPELINE ---

def run_pipeline():
    print("🚀 Starting Parquet Pipeline on HPC...")
    
    # 1. AGGREGATION
    print("🔨 Aggregating Tables...")
    agg_paths = []
    for group in DEPTH0_F:
        agg_paths.append(process_group(group, depth=0, prefix="train"))
    for group in DEPTH1_F:
        agg_paths.append(process_group(group, depth=1, prefix="train"))

    # 2. BATCHING
    print("💾 Creating Batches...")
    lf = pl.scan_parquet(os.path.join(BASE_DIR, f"train_{BASE_F}"))
    n_rows = lf.select(pl.len()).collect()[0, 0]
    batch_size = 200_000
    agg_lfs = [
    pl.scan_parquet(p).with_columns(pl.col("case_id").cast(pl.Int64)) 
    for p in agg_paths
]

    for offset in range(0, n_rows, batch_size):
        base_batch = lf.slice(offset, batch_size).collect()
        id_df = pl.DataFrame({"case_id": base_batch["case_id"]})
        
        for agg_lf in agg_lfs:
            feats = agg_lf.join(id_df.lazy(), on="case_id", how="inner").collect()
            base_batch = base_batch.join(feats, on="case_id", how="left")
            
        out_path = os.path.join(AGGR_DIR, f"batch_{offset:09d}.parquet")
        base_batch.write_parquet(out_path)
        print(f"   Wrote batch {offset} -> {out_path}")

    # 3. SPLIT (ROBUST TIME-SERIES)
    print("✂️ Splitting Data (Time-Series)...")
    paths = sorted(glob.glob(os.path.join(AGGR_DIR, "batch_*.parquet")))
    n_batches = len(paths)
    n_val = max(1, int(0.1 * n_batches))
    n_test = max(1, int(0.1 * n_batches))
    
    train_paths = paths[:-(n_val + n_test)]
    val_paths = paths[-(n_val + n_test):-n_test]
    test_paths = paths[-n_test:]
    
    print(f"   Train: {len(train_paths)} batches | Val: {len(val_paths)} batches | Test: {len(test_paths)} batches")

    # 4. TRAINING
    print("🏋️ Training LightGBM...")
    
    def load_batches(path_list):
        dfs = [pl.read_parquet(p) for p in path_list]
        return pl.concat(dfs, how="vertical")

    def df_to_xy(df):
        df_num = df.with_columns([
            pl.col(pl.Date, pl.Datetime).cast(pl.Int64),
            pl.col(pl.Boolean).cast(pl.Int8),
        ])
        X = df_num.select(pl.all().exclude(["target", "case_id"]).exclude(pl.Utf8)).to_numpy()
        y = df_num["target"].to_numpy()
        return X, y

    # Load Val Set
    val_df = load_batches(val_paths)
    X_val, y_val = df_to_xy(val_df)
    dvalid = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "auc",            # Metric fixed!
        "learning_rate": 0.03,      # Robust Param
        "num_leaves": 64,
        "colsample_bytree": 0.7,    # Robust Param
        "feature_pre_filter": False,
        "verbosity": -1,
        "n_jobs": -1
    }

    model = None
    best_score = 0
    no_improve = 0
    patience = 3

    for i, path in enumerate(train_paths):
        print(f"   Training on batch {i+1}/{len(train_paths)}: {os.path.basename(path)}")
        batch = pl.read_parquet(path)
        X_train_batch, y_train_batch = df_to_xy(batch)
        dtrain = lgb.Dataset(X_train_batch, label=y_train_batch, free_raw_data=False)

        if model is None:
            model = lgb.train(params, dtrain, num_boost_round=100, valid_sets=[dvalid], valid_names=["valid"], keep_training_booster=True)
        else:
            model = lgb.train(params, dtrain, init_model=model, num_boost_round=100, valid_sets=[dvalid], valid_names=["valid"], keep_training_booster=True)

        # FIXED KEYERROR: Using 'auc' because we set metric='auc'
        score = model.best_score["valid"]["auc"]
        print(f"      Valid AUC: {score:.5f}")

        if score > best_score + 1e-4:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("      Early stopping triggered.")
                break

    # 5. SAVE & EVALUATE
    print(f"💾 Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    print("📊 Evaluating Stability on Test Split...")
    test_df = load_batches(test_paths)
    X_test, y_test = df_to_xy(test_df)
    
    # Recover weeks for stability check
    weeks_test = test_df["WEEK_NUM"].to_numpy()
    
    y_pred = model.predict(X_test)
    
    stab_score, slope, _ = gini_stability(y_test, y_pred, weeks_test)
    auc_test = roc_auc_score(y_test, y_pred)
    
    print(f"\n✅ Final Test AUC:      {auc_test:.4f}")
    print(f"📉 Final Stability:     {stab_score:.4f}")
    print(f"📈 Final Slope:         {slope:.5f}")
    
    if slope >= 0:
        print("🎉 SUCCESS: Model is stable!")
    else:
        print("⚠️ WARNING: Model is degrading.")

if __name__ == "__main__":
    run_pipeline()