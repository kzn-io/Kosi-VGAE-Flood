# ================================================================
# KRB UNSUPERVISED FSM — STAGE 1: INPUT AUDIT + NODE ALIGNMENT
# Python 3.11 | Read-only with respect to existing project files
#
# Purpose:
#   1. Verify the existing 508-node feature matrix and V6 graph.
#   2. Inspect saved VGAE embeddings and discover Dink-style artifacts.
#   3. Report actual feature names so the FSM score can be physically
#      specified instead of guessing predictor directionality.
#   4. Write an audit report to the FSM output directory.
#
# This stage does NOT train on FIM, alter regionalization, or assign
# susceptibility scores. Those require the verified feature semantics.
# ================================================================

from pathlib import Path
import traceback
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(r"D:\KRB_FINAL_GRAPH_PROJECT_V6")
FEATURE_CSV = Path(
    r"D:\KRB\KRB-DATASET\Harmonized_30m"
    r"\KRB_STAGE2_FEATURES_FROM_FINAL_INPUTS"
    r"\scaled_rasters\KRB_508_Basin_Feature_Matrix.csv"
)
GRAPH_PT = ROOT / "spatial_graph.pt"
NODE_MAPPING_CSV = ROOT / "node_mapping.csv"
VGAE_DIR = ROOT / "VGAE_BASELINE"
VGAE_EMBEDDINGS = VGAE_DIR / "V6_VGAE_embeddings.npy"
DINK_DIR = ROOT / "DINKNET_STYLE"
OUT_DIR = ROOT / "FSM_UNSUPERVISED"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "FSM_STAGE1_INPUT_AUDIT.txt"

lines = []

def log(s=""):
    s = str(s)
    print(s)
    lines.append(s)

def describe_array(path):
    arr = np.load(path, allow_pickle=False)
    log(f"  array shape: {arr.shape}")
    log(f"  dtype: {arr.dtype}")
    log(f"  all finite: {bool(np.isfinite(arr).all())}")
    if arr.ndim >= 1:
        log(f"  first dimension: {arr.shape[0]}")
    return arr

log("=" * 76)
log("KRB UNSUPERVISED FSM — STAGE 1 INPUT AUDIT")
log("=" * 76)
log("Existing regionalization is being reused; it will not be rerun.")
log("FIM is reserved for external validation and is not used here.")
log(f"Project root: {ROOT}")
log(f"Output directory: {OUT_DIR}")

# 1. Feature matrix
features = None
if FEATURE_CSV.exists():
    log("\n[1] FEATURE MATRIX")
    features = pd.read_csv(FEATURE_CSV)
    log(f"Path: {FEATURE_CSV}")
    log(f"Shape: {features.shape}")
    log(f"Columns: {features.columns.tolist()}")
    log(f"Duplicate column names: {features.columns.duplicated().any()}")
    log(f"Duplicate rows: {int(features.duplicated().sum())}")

    possible_ids = [
        c for c in features.columns
        if c.lower() in {"basin_id", "micro_id", "catchment_id",
                         "subbasin_id", "node_id", "id"}
        or c.lower().endswith("_id")
    ]
    log(f"Possible ID columns: {possible_ids}")

    numeric = features.select_dtypes(include=[np.number])
    nonconstant = [
        c for c in numeric.columns
        if numeric[c].nunique(dropna=False) > 1
    ]
    constant = [c for c in numeric.columns if c not in nonconstant]
    log(f"Numeric columns: {len(numeric.columns)}")
    log(f"Nonconstant numeric columns: {len(nonconstant)}")
    log(f"Constant numeric columns: {constant}")
    log(f"Missing numeric values: {int(numeric.isna().sum().sum())}")
    if len(numeric.columns):
        log(f"Infinite numeric values: {int(np.isinf(numeric.to_numpy()).sum())}")
    log("NONCONSTANT NUMERIC COLUMN NAMES:")
    for c in nonconstant:
        log(f"  - {c}")
else:
    log(f"\n[1] MISSING FEATURE MATRIX: {FEATURE_CSV}")

# 2. Node mapping
mapping = None
if NODE_MAPPING_CSV.exists():
    log("\n[2] NODE MAPPING")
    mapping = pd.read_csv(NODE_MAPPING_CSV)
    log(f"Path: {NODE_MAPPING_CSV}")
    log(f"Shape: {mapping.shape}")
    log(f"Columns: {mapping.columns.tolist()}")
    log(mapping.head(8).to_string(index=False))
else:
    log(f"\n[2] MISSING NODE MAPPING: {NODE_MAPPING_CSV}")

# 3. Graph
if GRAPH_PT.exists():
    log("\n[3] V6 SPATIAL GRAPH")
    log(f"Path: {GRAPH_PT}")
    if torch is None:
        log("PyTorch is not installed; graph object could not be inspected.")
    else:
        try:
            obj = torch.load(Relative path	Bytes
.gitignore	215
01_raster_harmonization/01_align_environmental_rasters_to_reference_grid.py	9115
01_raster_harmonization/02_derive_terrain_and_hydrological_predictors.py	33687
02_feature_engineering/03_construct_microcatchment_predictor_features.py	33687
03_quality_control/04_audit_flood_susceptibility_model_inputs.py	7893
03_quality_control/05_audit_vgae_embeddings_and_fsm_inputs.py	7893
03_quality_control/06_validate_feature_geometry_and_graph_alignment.py	1874
04_graph_construction/07_construct_microcatchment_spatial_adjacency.py	6194
04_graph_construction/08_build_508_node_spatial_graph.py	6194
05_vgae_and_clustering/09_generate_vgae_five_class_susceptibility.py	6126
05_vgae_and_clustering/10_run_unsupervised_susceptibility_regionalization.py	11330
05_vgae_and_clustering/11_evaluate_embedding_clusters.py	1226
06_surrogate_and_shap/KRB_SHAP_STEP01_DATA_QC.py	9749
06_surrogate_and_shap/KRB_SHAP_STEP02_DATA_SPLIT.py	7567
06_surrogate_and_shap/KRB_SHAP_STEP03_SURROGATE_MODELS_REVISED.py	11469
06_surrogate_and_shap/KRB_SHAP_STEP04_MULTIMODEL_SHAP.py	18856
06_surrogate_and_shap/KRB_SHAP_STEP05F_BRIGHT_COLOR_RAMP.py	23815
07_spatial_explainability/KRB_SPATIAL_SHAP_CLASS5_ATLAS_30x18_LARGE_FONTS_PATHFIX.py	22806
MANIFEST.txt	3697
README.md	9209
data/README.md	787
data/flood_susceptibility/vgae_dinknet_fsm_comparison_layers_508.gpkg	22114304
data/flood_susceptibility/vgae_flood_susceptibility_5class_508.gpkg	5595136
data/graph/KRB_Candidate_Adjacency.npz	6179
data/graph/KRB_Candidate_Adjacency_Edges.csv	25225
data/graph/KRB_Directed_Adjacency.npz	6086
data/graph/KRB_Directed_Hydrological_Edges.csv	32251
data/graph/KRB_Graph_Node_Order.csv	4375
data/graph_construction/README.md	2633
data/graph_construction/node_features/microcatchment_feature_matrix_508.csv	335148
data/graph_construction/outlet_routing_validation/microcatchment_outlet_summary.csv	17292
data/graph_construction/outlet_routing_validation/outlet_trace_edge_diagnostics.csv	32262
data/graph_construction/spatial_units/krb_microcatchment_features_508.gpkg	5726208
data/graph_construction/topology/candidate_adjacency_matrix_508.npz	6179
data/graph_construction/topology/candidate_spatial_adjacency_edges.csv	25225
data/graph_construction/topology/directed_hydrological_adjacency_508.npz	6086
data/graph_construction/topology/directed_hydrological_edges.csv	32251
data/graph_construction/topology/graph_node_order_508.csv	4375
data/processed/Class5_Spatial_SHAP_Values_508_Microcatchments.csv	195312
data/processed/KRB_508_Basin_Feature_Matrix.csv	335148
data/processed/LightGBM_SHAP_LONG_ALL_508.csv	6469794
data/spatial_explainability/README.md	946
data/spatial_explainability/class5_spatial_shap_explanations_508.gpkg	5681152
data/spatial_explainability/class5_top_six_predictor_values_508.csv	195312
data/spatial_explainability/class5_top_six_shap_feature_ranking.csv	342
data/spatial_explainability/lightgbm_shap_values_long_all_classes_508.csv	6469794
data/spatial_units/krb_microcatchment_boundaries_508.cpg	5
data/spatial_units/krb_microcatchment_boundaries_508.dbf	53058
data/spatial_units/krb_microcatchment_boundaries_508.prj	401
data/spatial_units/krb_microcatchment_boundaries_508.shp	5060924
data/spatial_units/krb_microcatchment_boundaries_508.shx	4164
data/spatial_units/krb_microcatchment_feature_attributes_508.gpkg	5726208
docs/DATA_DICTIONARY.md	2220
docs/LOCAL_PATH_CONFIGURATION.md	648
docs/PIPELINE_AND_EXECUTION.md	3652
docs/REPRODUCIBILITY_GAPS.md	2189
requirements.txt	128
results/shap_analysis/Global_SHAP_Mean_Absolute_Importance.csv	2393
results/shap_analysis/KRB_SHAP_STEP03_CV_RESULTS.csv	1331
results/shap_analysis/KRB_SHAP_STEP03_MODEL_METRICS.csv	803
results/shap_analysis/KRB_SHAP_STEP04_LightGBM_CLASS_5_BEESWARM.png	582830
results/shap_analysis/KRB_SHAP_