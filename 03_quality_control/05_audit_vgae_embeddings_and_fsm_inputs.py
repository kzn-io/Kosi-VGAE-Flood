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
            obj = torch.load(GRAPH_PT, map_location="cpu", weights_only=False)
            log(f"Object type: {type(obj)}")
            if hasattr(obj, "num_nodes"):
                log(f"Graph num_nodes: {obj.num_nodes}")
            if hasattr(obj, "edge_index"):
                ei = obj.edge_index.detach().cpu().numpy()
                log(f"edge_index shape: {ei.shape}")
                if ei.ndim == 2 and ei.shape[0] == 2:
                    log(f"Minimum node index: {int(ei.min())}")
                    log(f"Maximum node index: {int(ei.max())}")
                    log(f"Directed edge entries: {ei.shape[1]}")
            elif isinstance(obj, dict):
                log(f"Dictionary keys: {list(obj.keys())}")
                for k, v in obj.items():
                    if hasattr(v, "shape"):
                        log(f"  {k}: shape={tuple(v.shape)}")
        except Exception as e:
            log(f"Graph load/inspection error: {repr(e)}")
            log(traceback.format_exc())
else:
    log(f"\n[3] MISSING V6 GRAPH: {GRAPH_PT}")

# 4. VGAE embedding
if VGAE_EMBEDDINGS.exists():
    log("\n[4] VGAE EMBEDDINGS")
    log(f"Path: {VGAE_EMBEDDINGS}")
    try:
        vgae = describe_array(VGAE_EMBEDDINGS)
        if features is not None and vgae.shape[0] != len(features):
            log("WARNING: VGAE embedding row count does not match feature rows.")
    except Exception as e:
        log(f"Embedding inspection error: {repr(e)}")
else:
    log(f"\n[4] MISSING VGAE EMBEDDINGS: {VGAE_EMBEDDINGS}")

# 5. Discover Dink artifacts
log("\n[5] DINK-NET-INSPIRED OUTPUT INVENTORY")
if DINK_DIR.exists():
    artifact_count = 0
    for p in sorted(DINK_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in {
            ".npy", ".pt", ".pth", ".csv", ".gpkg", ".json"
        }:
            log(f"  {p}")
            artifact_count += 1
            if p.suffix.lower() == ".npy":
                try:
                    a = np.load(p, allow_pickle=False, mmap_mode="r")
                    log(f"      shape={a.shape}, dtype={a.dtype}")
                except Exception as e:
                    log(f"      array inspection skipped: {repr(e)}")
    log(f"Total discovered artifacts: {artifact_count}")
else:
    log(f"Dink directory not found: {DINK_DIR}")

# 6. Feature name keyword inventory to guide the physically signed index
if features is not None:
    log("\n[6] CANDIDATE FLOOD-CONDITIONING FEATURES (NAME MATCH ONLY)")
    terms = [
        "elev", "dem", "height", "slope", "aspect", "curv", "twi",
        "spi", "sti", "flow", "accum", "drain", "stream", "river",
        "rain", "precip", "soil", "clay", "silt", "sand", "land",
        "cover", "imperv", "ndvi", "water", "distance", "dist",
        "hand", "wetness", "rough", "geomorph"
    ]
    matched = [
        c for c in features.columns
        if any(term in c.lower() for term in terms)
    ]
    for c in matched:
        log(f"  - {c}")
    if not matched:
        log("No candidate names matched; inspect full predictor list above.")
    log("IMPORTANT: keyword matches do not establish the correct sign or physical meaning.")

# 7. Alignment summary
log("\n[7] ALIGNMENT CHECK SUMMARY")
if features is not None:
    log(f"Feature rows: {len(features)}")
if mapping is not None:
    log(f"Mapping rows: {len(mapping)}")
    if features is not None:
        log(f"Feature/mapping row count equal: {len(features) == len(mapping)}")
log("Do not assume row order is aligned unless the ID columns are compared explicitly.")
log("\nNEXT: use this report to lock the ID mapping and feature semantics before scoring.")
log("No model training, clustering, FIM access, or susceptibility scoring occurred.")
log("=" * 76)

REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nAudit report saved to: {REPORT}")
