# ================================================================
# KRB VGAE-SHAP | STEP 1: DATA LOADING, ALIGNMENT AND QC
# ================================================================
# Purpose:
#   1. Load the 508-basin environmental feature matrix.
#   2. Load and verify the final VGAE five-class labels.
#   3. Confirm basin_id = node_index + 1 alignment.
#   4. Cross-check labels against the combined VGAE/DinkNet GeoPackage.
#   5. Remove identifier and zero-variance columns from predictors.
#   6. Export clean modelling table, aligned spatial layer and QC report.
#
# This step DOES NOT train models or modify the original map.
# ================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------- PATHS ----------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "shap_analysis"

FEATURE_CSV = DATA / "processed" / "KRB_508_Basin_Feature_Matrix.csv"
ALL_LAYERS_GPKG = DATA / "flood_susceptibility" / "vgae_dinknet_fsm_comparison_layers_508.gpkg"
FINAL_CLASS_GPKG = DATA / "flood_susceptibility" / "vgae_flood_susceptibility_5class_508.gpkg"
FINAL_CLASS_LAYER = "vgae_fsm_5class"

OUTPUT_DIR = RESULTS / "step01_data_qc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLE = OUTPUT_DIR / "clean_modelling_data_508.csv"
OUT_GPKG = OUTPUT_DIR / "aligned_microcatchments_508.gpkg"
OUT_REPORT = OUTPUT_DIR / "qc_report.txt"

# ------------------------- HELPERS --------------------------------

def stop(message):
    raise RuntimeError("\nDATA QC FAILED:\n" + str(message))


def write_report(lines):
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


# ------------------------- CHECK FILES ----------------------------

for path in [FEATURE_CSV, ALL_LAYERS_GPKG, FINAL_CLASS_GPKG]:
    if not path.exists():
        stop(f"Input file not found: {path}")

report = []
report.append("KRB VGAE-SHAP | STEP 1 DATA QC")
report.append("=" * 65)

# ------------------------- READ CSV ------------------------------

features = pd.read_csv(FEATURE_CSV)

if "basin_id" not in features.columns:
    stop("CSV does not contain the required basin_id field.")

if features["basin_id"].duplicated().any():
    stop("Duplicate basin_id values found in feature matrix.")

if len(features) != 508:
    stop(f"Expected 508 feature rows; found {len(features)}.")

# Require numeric predictor columns.
predictor_candidates = [c for c in features.columns if c != "basin_id"]

non_numeric = [
    c for c in predictor_candidates
    if not pd.api.types.is_numeric_dtype(features[c])
]
if non_numeric:
    stop(f"Non-numeric predictor columns found: {non_numeric}")

X_all = features[predictor_candidates].copy()

# Missing and infinite value checks.
missing_count = int(X_all.isna().sum().sum())
infinite_count = int(np.isinf(X_all.to_numpy(dtype=float)).sum())

if missing_count > 0:
    stop(f"Feature matrix contains {missing_count} missing values.")
if infinite_count > 0:
    stop(f"Feature matrix contains {infinite_count} infinite values.")

# Identify zero-variance predictors.
nunique = X_all.nunique(dropna=False)
constant_features = nunique[nunique <= 1].index.tolist()
usable_features = [c for c in predictor_candidates if c not in constant_features]

if len(usable_features) == 0:
    stop("No usable predictors remain after constant-column check.")

# ------------------------- READ FINAL MAP -------------------------

final_gdf = gpd.read_file(
    FINAL_CLASS_GPKG,
    layer=FINAL_CLASS_LAYER
)

required_fields = {
    "node_index", "VGAE_FSM5", "VGAE_FSM5_name",
    "VGAE_FSM_cluster10"
}
missing_fields = required_fields.difference(final_gdf.columns)

if missing_fields:
    stop(f"Final class layer is missing fields: {sorted(missing_fields)}")

if len(final_gdf) != 508:
    stop(f"Expected 508 final polygons; found {len(final_gdf)}.")

if final_gdf["node_index"].duplicated().any():
    stop("Duplicate node_index values found in final class layer.")

# The expected node indices are 0..507.
expected_nodes = set(range(508))
actual_nodes = set(pd.to_numeric(final_gdf["node_index"]).astype(int))

if actual_nodes != expected_nodes:
    stop("node_index values are not exactly the expected range 0..507.")

# ------------------------- ALIGN IDs ------------------------------

features["basin_id"] = pd.to_numeric(
    features["basin_id"], errors="raise"
).astype(int)

final_gdf["node_index"] = pd.to_numeric(
    final_gdf["node_index"], errors="raise"
).astype(int)

# Required one-to-one relation:
# CSV basin_id 1..508 <-> GeoPackage node_index 0..507
final_gdf["basin_id"] = final_gdf["node_index"] + 1

if set(features["basin_id"]) != set(final_gdf["basin_id"]):
    stop("CSV basin_id and GeoPackage node_index+1 do not match.")

# Sort and join by explicit basin_id.
features = features.sort_values("basin_id").reset_index(drop=True)
final_gdf = final_gdf.sort_values("basin_id").reset_index(drop=True)

if not np.array_equal(
    features["basin_id"].to_numpy(),
    final_gdf["basin_id"].to_numpy()
):
    stop("Basin alignment failed after sorting.")

# ------------------------- LABEL CHECKS ---------------------------

final_gdf["VGAE_FSM5"] = pd.to_numeric(
    final_gdf["VGAE_FSM5"], errors="raise"
).astype(int)

allowed_classes = {1, 2, 3, 4, 5}
observed_classes = set(final_gdf["VGAE_FSM5"].unique())

if observed_classes != allowed_classes:
    stop(
        f"Expected classes {allowed_classes}; "
        f"found {observed_classes}."
    )

class_counts = (
    final_gdf["VGAE_FSM5"]
    .value_counts()
    .sort_index()
    .to_dict()
)

expected_counts = {1: 117, 2: 103, 3: 100, 4: 103, 5: 85}
if class_counts != expected_counts:
    stop(
        "Five-class distribution differs from the previously verified "
        f"distribution. Found: {class_counts}"
    )

# Cross-check the same labels in the combined GeoPackage.
all_layers = gpd.read_file(
    ALL_LAYERS_GPKG,
    layer=FINAL_CLASS_LAYER
)

if not {"node_index", "VGAE_FSM5"}.issubset(all_layers.columns):
    stop("Combined GeoPackage lacks node_index or VGAE_FSM5.")

if len(all_layers) != 508:
    stop(
        "Combined GeoPackage vgae_fsm_5class layer does not contain "
        f"508 features; found {len(all_layers)}."
    )

comparison = final_gdf[["node_index", "VGAE_FSM5"]].merge(
    all_layers[["node_index", "VGAE_FSM5"]],
    on="node_index",
    how="outer",
    suffixes=("_standalone", "_combined"),
    validate="one_to_one",
    indicator=True
)

if not (comparison["_merge"] == "both").all():
    stop("Node sets differ between standalone and combined GeoPackages.")

if not (
    comparison["VGAE_FSM5_standalone"]
    == comparison["VGAE_FSM5_combined"]
).all():
    stop("VGAE_FSM5 labels differ between the two GeoPackages.")

# ------------------------- BUILD MODEL TABLE ----------------------

# Keep only the 34 non-constant predictors.
model_table = features[["basin_id"] + usable_features].merge(
    final_gdf[[
        "basin_id", "node_index", "VGAE_FSM5", "VGAE_FSM5_name"
    ]],
    on="basin_id",
    how="inner",
    validate="one_to_one"
)

if len(model_table) != 508:
    stop(f"Final modelling table has {len(model_table)} rows, not 508.")

# Keep identifiers and target separate from predictors.
# The CSV output includes identifiers/target for traceability;
# Step 2 will explicitly select only usable_features as X.
model_table.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

# Add basin_id to spatial layer and export a working copy.
# Original input GeoPackages remain untouched.
spatial_output = final_gdf.copy()
spatial_output.to_file(
    OUT_GPKG,
    layer="krb_shap_step01_basins",
    driver="GPKG"
)

# ------------------------- REPORT ---------------------------------

report.extend([
    f"Feature matrix rows: {len(features)}",
    f"Original CSV columns: {features.shape[1]}",
    f"Predictor candidates: {len(predictor_candidates)}",
    f"Zero-variance predictors removed: {len(constant_features)}",
    f"Usable predictors: {len(usable_features)}",
    f"Missing values: {missing_count}",
    f"Infinite values: {infinite_count}",
    f"Final polygons: {len(final_gdf)}",
    f"CRS: {final_gdf.crs}",
    "ID alignment: basin_id = node_index + 1",
    "Standalone/combined class labels: IDENTICAL",
    "",
    "ZERO-VARIANCE FEATURES:",
    *(constant_features if constant_features else ["None"]),
    "",
    "CLASS COUNTS:",
    *(f"Class {k}: {v}" for k, v in class_counts.items()),
    "",
    "USABLE PREDICTORS:",
    *usable_features,
    "",
    "OUTPUTS:",
    str(OUT_TABLE),
    str(OUT_GPKG),
    str(OUT_REPORT),
    "",
    "STATUS: PASS"
])

write_report(report)

print("\n" + "=" * 65)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 65)
print(f"Observations             : {len(model_table)}")
print(f"Usable predictors        : {len(usable_features)}")
print(f"Zero-variance removed    : {len(constant_features)}")
print(f"Class counts             : {class_counts}")
print(f"CRS                      : {final_gdf.crs}")
print("\nRemoved constant features:")
for col in constant_features:
    print(" -", col)
print("\nSaved outputs:")
print(OUT_TABLE)
print(OUT_GPKG)
print(OUT_REPORT)
print("\nNo model was trained and no original map was modified.")
