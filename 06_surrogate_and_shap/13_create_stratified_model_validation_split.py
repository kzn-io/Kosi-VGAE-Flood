# ================================================================
# KRB VGAE-SHAP | STEP 2: SURROGATE DATASET + HOLD-OUT DESIGN
# ================================================================
# Run ONLY after STEP 1 has completed successfully.
#
# Purpose:
#   1. Read the QC-approved modelling table from Step 1.
#   2. Define the 34 environmental predictors and VGAE_FSM5 target.
#   3. Create a reproducible stratified 80:20 train/test split.
#   4. Save split datasets and a split/QC report.
#
# Important:
#   - No model training in this step.
#   - No feature scaling is applied: tree models do not require it.
#   - basin_id and node_index are identifiers, never predictors.
#   - Test data must remain untouched until final model evaluation.
# ================================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------- PATHS ----------------------------------
ROOT = Path(__file__).resolve().parents[1]
STEP1_DIR = ROOT / "results" / "shap_analysis" / "step01_data_qc"
INPUT_TABLE = STEP1_DIR / "clean_modelling_data_508.csv"
STEP1_REPORT = STEP1_DIR / "qc_report.txt"
OUTPUT_DIR = ROOT / "results" / "shap_analysis" / "step02_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CSV = OUTPUT_DIR / "train_80pct.csv"
TEST_CSV = OUTPUT_DIR / "test_20pct.csv"
FEATURES_JSON = OUTPUT_DIR / "predictors.json"
REPORT_TXT = OUTPUT_DIR / "split_report.txt"

# ------------------------- SETTINGS ------------------------------

TARGET = "VGAE_FSM5"
ID_COLUMNS = ["basin_id", "node_index"]
TEST_SIZE = 0.20
RANDOM_STATE = 42
EXPECTED_ROWS = 508
EXPECTED_CLASSES = [1, 2, 3, 4, 5]

# ------------------------- VALIDATE INPUT -------------------------

if not INPUT_TABLE.exists():
    raise FileNotFoundError(
        f"Step 1 output not found:\n{INPUT_TABLE}\n"
        "Run STEP 1 first and confirm that it finishes with STATUS: PASS."
    )

if not STEP1_REPORT.exists():
    raise FileNotFoundError(
        f"Step 1 QC report not found:\n{STEP1_REPORT}"
    )

qc_text = STEP1_REPORT.read_text(encoding="utf-8")
if "STATUS: PASS" not in qc_text:
    raise RuntimeError(
        "Step 1 report does not contain STATUS: PASS. "
        "Do not proceed until Step 1 passes."
    )

df = pd.read_csv(INPUT_TABLE)

required = set(ID_COLUMNS + [TARGET])
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Required fields missing from Step 1 table: {sorted(missing)}")

if len(df) != EXPECTED_ROWS:
    raise ValueError(f"Expected {EXPECTED_ROWS} rows; found {len(df)}.")

if df["basin_id"].duplicated().any():
    raise ValueError("Duplicate basin_id values found.")

if df["node_index"].duplicated().any():
    raise ValueError("Duplicate node_index values found.")

# Verify identifier relationship again.
if not np.array_equal(
    df["basin_id"].to_numpy(dtype=int),
    df["node_index"].to_numpy(dtype=int) + 1
):
    raise ValueError("ID alignment failed: basin_id must equal node_index + 1.")

# ------------------------- DEFINE X AND y -------------------------

excluded = set(ID_COLUMNS + [TARGET, "VGAE_FSM5_name"])
predictor_columns = [c for c in df.columns if c not in excluded]

if len(predictor_columns) != 34:
    raise ValueError(
        f"Expected 34 predictors after Step 1; found {len(predictor_columns)}. "
        "Review the Step 1 QC report before proceeding."
    )

X = df[predictor_columns].copy()
y = pd.to_numeric(df[TARGET], errors="raise").astype(int)

if y.isna().any():
    raise ValueError("Target contains missing labels.")

if sorted(y.unique().tolist()) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected target classes {EXPECTED_CLASSES}; "
        f"found {sorted(y.unique().tolist())}."
    )

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Predictor matrix contains NaN or infinite values.")

# ------------------------- STRATIFIED SPLIT -----------------------

row_indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    row_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

train_df = df.iloc[train_idx].copy().sort_values("basin_id")
test_df = df.iloc[test_idx].copy().sort_values("basin_id")

# Confirm no overlap.
train_ids = set(train_df["basin_id"])
test_ids = set(test_df["basin_id"])

if train_ids.intersection(test_ids):
    raise RuntimeError("Train/test basin overlap detected.")

if train_ids.union(test_ids) != set(df["basin_id"]):
    raise RuntimeError("Train/test split does not cover all basins.")

# Verify every class occurs in both subsets.
train_counts = train_df[TARGET].value_counts().sort_index()
test_counts = test_df[TARGET].value_counts().sort_index()

if set(train_counts.index) != set(EXPECTED_CLASSES):
    raise RuntimeError("At least one target class is absent from training data.")

if set(test_counts.index) != set(EXPECTED_CLASSES):
    raise RuntimeError("At least one target class is absent from test data.")

# ------------------------- SAVE SPLITS ----------------------------

train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
test_df.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")

FEATURES_JSON.write_text(
    json.dumps(
        {
            "target": TARGET,
            "predictors": predictor_columns,
            "excluded_identifier_columns": ID_COLUMNS,
            "excluded_non_predictor_columns": ["VGAE_FSM5_name"],
            "n_predictors": len(predictor_columns),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "split_method": "stratified random hold-out"
        },
        indent=4
    ),
    encoding="utf-8"
)

# ------------------------- REPORT --------------------------------

lines = [
    "KRB VGAE-SHAP | STEP 2 SPLIT REPORT",
    "=" * 65,
    f"Total basins: {len(df)}",
    f"Predictor count: {len(predictor_columns)}",
    f"Target: {TARGET}",
    f"Training basins: {len(train_df)}",
    f"Testing basins: {len(test_df)}",
    f"Test fraction: {TEST_SIZE:.2f}",
    f"Random state: {RANDOM_STATE}",
    "Split method: Stratified random hold-out",
    "Identifier leakage check: PASS",
    "Train/test overlap check: PASS",
    "",
    "FULL DATA CLASS COUNTS:",
    *(f"Class {int(k)}: {int(v)}" for k, v in y.value_counts().sort_index().items()),
    "",
    "TRAIN CLASS COUNTS:",
    *(f"Class {int(k)}: {int(v)}" for k, v in train_counts.items()),
    "",
    "TEST CLASS COUNTS:",
    *(f"Class {int(k)}: {int(v)}" for k, v in test_counts.items()),
    "",
    "PREDICTORS:",
    *predictor_columns,
    "",
    "OUTPUTS:",
    str(TRAIN_CSV),
    str(TEST_CSV),
    str(FEATURES_JSON),
    str(REPORT_TXT),
    "",
    "STATUS: PASS"
]

REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("\n" + "=" * 65)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 65)
print(f"Total basins       : {len(df)}")
print(f"Predictors         : {len(predictor_columns)}")
print(f"Training basins    : {len(train_df)}")
print(f"Testing basins     : {len(test_df)}")
print(f"Train class counts : {train_counts.to_dict()}")
print(f"Test class counts  : {test_counts.to_dict()}")
print("\nSaved:")
print(TRAIN_CSV)
print(TEST_CSV)
print(FEATURES_JSON)
print(REPORT_TXT)
print("\nNo model was trained in this step.")
