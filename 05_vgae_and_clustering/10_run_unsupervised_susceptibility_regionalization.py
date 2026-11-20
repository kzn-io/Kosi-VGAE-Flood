# ======================================================================
# KRB UNSUPERVISED FSM — STAGE 2
# Physically anchored relative susceptibility index + SSL representation
# zoning using EXISTING VGAE and Dink-Net-inspired embeddings.
#
# Uses:
#   - 508-basin feature matrix
#   - V6 node mapping and spatial graph
#   - existing VGAE embeddings
#   - existing Dink-inspired embeddings (K16 seeds 11/22/33)
#
# IMPORTANT:
#   * No FIM raster or flood labels are read.
#   * No regionalization labels are reused as flood labels.
#   * Scores are RELATIVE INDEX values, not calibrated flood probabilities.
#   * Factor directions/weights are explicit and saved for auditability.
#   * Embedding-zone KMeans is exploratory; zones are ranked using the
#     independent physical index, not by arbitrary cluster number.
# ======================================================================

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import spearmanr

ROOT = Path(r"D:\KRB_FINAL_GRAPH_PROJECT_V6")
FEATURE_CSV = Path(
    r"D:\KRB\KRB-DATASET\Harmonized_30m"
    r"\KRB_STAGE2_FEATURES_FROM_FINAL_INPUTS"
    r"\scaled_rasters\KRB_508_Basin_Feature_Matrix.csv"
)
MAPPING_CSV = ROOT / "node_mapping.csv"
GRAPH_PT = ROOT / "spatial_graph.pt"
VGAE_EMB = ROOT / "VGAE_BASELINE" / "V6_VGAE_embeddings.npy"
DINK_DIR = ROOT / "DINKNET_STYLE"
OUT = ROOT / "FSM_UNSUPERVISED"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_ZONES = 5  # exploratory susceptibility strata in embedding space

# Physically directional factors. Each is converted to percentile rank [0,1]
# so raw scale differences do not dominate. Directions are explicit.
# Higher values after sign correction mean "more flood-conditioning".
FACTORS = {
    "DEM_scaled_mean": -1,                    # lower elevation -> higher
    "Slope_scaled_mean": -1,                  # lower slope -> higher
    "Distance_to_Main_River_scaled_mean": -1, # closer to main river -> higher
    "Drainage_Density_scaled_mean": +1,       # greater drainage density -> higher
    "Max1Day_Rainfall_scaled_mean": +1,        # larger extreme daily rainfall -> higher
    "Mean_Rainfall_scaled_mean": +1,           # greater mean rainfall -> higher
    "TWI_scaled_mean": +1,                     # wetter topographic setting -> higher
}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def percentile_score(series, direction):
    # Average rank handles ties; rank-normalized score is robust to input units.
    ranks = series.rank(method="average", pct=True)
    return ranks if direction > 0 else 1.0 - ranks

def load_embedding(path):
    arr = np.load(path, allow_pickle=False)
    require(arr.ndim == 2, f"Embedding must be 2D: {path}, shape={arr.shape}")
    require(np.isfinite(arr).all(), f"Non-finite values in embedding: {path}")
    return arr.astype(np.float64, copy=False)

def cluster_embedding(arr, k, seed):
    z = StandardScaler().fit_transform(arr)
    km = KMeans(n_clusters=k, random_state=seed, n_init=50)
    labels = km.fit_predict(z)
    sil = float(silhouette_score(z, labels)) if len(np.unique(labels)) > 1 else np.nan
    return labels, sil

print("=" * 72)
print("KRB FSM STAGE 2 — UNSUPERVISED RELATIVE SUSCEPTIBILITY")
print("=" * 72)
print("FIM is NOT read or used. This script does not train on flood labels.")

# Load table and node mapping
df = pd.read_csv(FEATURE_CSV)
mapping = pd.read_csv(MAPPING_CSV)
require(len(df) == 508, f"Expected 508 feature rows, found {len(df)}")
require(len(mapping) == 508, f"Expected 508 node rows, found {len(mapping)}")
require("basin_id" in df.columns, "Feature CSV missing basin_id")
require({"node_index", "feature_id"}.issubset(mapping.columns),
        "Node mapping needs node_index and feature_id columns")
require(mapping["node_index"].is_unique, "node_index is not unique")
require(mapping["feature_id"].is_unique, "feature_id is not unique")
require(df["basin_id"].is_unique, "basin_id is not unique")

# Explicit node order: map feature_id to basin_id. Existing project indicates
# these IDs match; verify before relying on row order.
id_to_row = {int(bid): i for i, bid in enumerate(df["basin_id"])}
map_ids = mapping["feature_id"].astype(int).tolist()
missing = [x for x in map_ids if x not in id_to_row]
require(not missing, f"Mapping feature_id values absent from basin_id: {missing[:10]}")
mapping = mapping.sort_values("node_index").reset_index(drop=True)
ordered_rows = [id_to_row[int(x)] for x in mapping["feature_id"]]
node_df = df.iloc[ordered_rows].reset_index(drop=True)
require(np.array_equal(node_df["basin_id"].astype(int).to_numpy(),
                       mapping["feature_id"].astype(int).to_numpy()),
        "Node-to-feature ID alignment failed")

print(f"Verified nodes: {len(node_df)}")
print("Node mapping: aligned by feature_id == basin_id")

# Physical index from named predictors
missing_factors = [c for c in FACTORS if c not in node_df.columns]
require(not missing_factors, f"Missing required factors: {missing_factors}")

components = pd.DataFrame({"basin_id": node_df["basin_id"].astype(int)})
for col, direction in FACTORS.items():
    x = pd.to_numeric(node_df[col], errors="coerce")
    require(x.notna().all(), f"Missing/non-numeric values in {col}")
    require(np.isfinite(x.to_numpy()).all(), f"Non-finite values in {col}")
    components[col + "_susceptibility_rank"] = percentile_score(x, direction)

component_cols = [c for c in components if c.endswith("_susceptibility_rank")]
components["FSM_physical_index"] = components[component_cols].mean(axis=1)
components["FSM_physical_rank_pct"] = components["FSM_physical_index"].rank(
    method="average", pct=True
)
components["FSM_zone_5class"] = pd.qcut(
    components["FSM_physical_rank_pct"],
    q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
).astype(int)

# Weight sensitivity: omit one factor at a time; this is a diagnostic only.
sensitivity_rows = []
base = components["FSM_physical_index"].to_numpy()
for col in component_cols:
    subset = [c for c in component_cols if c != col]
    alt = components[subset].mean(axis=1).to_numpy()
    rho = spearmanr(base, alt).statistic
    sensitivity_rows.append({
        "omitted_component": col,
        "spearman_vs_full_index": float(rho),
        "top_decile_overlap_count": int(
            len(set(np.argsort(base)[-max(1, int(np.ceil(len(base)*0.10))):]) &
                    set(np.argsort(alt)[-max(1, int(np.ceil(len(alt)*0.10))):]))
        )
    })
pd.DataFrame(sensitivity_rows).to_csv(
    OUT / "FSM_factor_sensitivity.csv", index=False
)

# Load SSL embeddings and derive exploratory zones in latent space.
# These zones are not themselves ranked as flood classes until summarized
# against the independently defined physical index.
embeddings = {}
embeddings["VGAE"] = load_embedding(VGAE_EMB)
require(embeddings["VGAE"].shape[0] == len(node_df),
        "VGAE embedding rows do not equal aligned node count")

for seed in (11, 22, 33):
    p = DINK_DIR / f"K16_seed{seed}_embeddings.npy"
    require(p.exists(), f"Missing Dink K16 embedding: {p}")
    embeddings[f"DINK_K16_seed{seed}"] = load_embedding(p)
    require(embeddings[f"DINK_K16_seed{seed}"].shape[0] == len(node_df),
            f"Dink embedding row count mismatch: {p}")

# Combine physical index with embedding zones in the final per-basin table.
results = components.copy()
zone_summary = []
for name, emb in embeddings.items():
    labels, sil = cluster_embedding(emb, N_ZONES, SEED)
    raw_col = f"{name}_latent_cluster"
    results[raw_col] = labels + 1  # display labels 1..K, still arbitrary

    tmp = pd.DataFrame({
        "cluster": labels,
        "index": components["FSM_physical_index"].to_numpy(),
        "rank_pct": components["FSM_physical_rank_pct"].to_numpy()
    })
    means = tmp.groupby("cluster")["index"].mean().sort_values()
    rank_map = {int(cluster): rank + 1
                for rank, cluster in enumerate(means.index)}
    ranked_col = f"{name}_zone_rank_by_physical_index"
    results[ranked_col] = [rank_map[int(x)] for x in labels]

    for cluster, group in tmp.groupby("cluster"):
        zone_summary.append({
            "representation": name,
            "latent_cluster_original_label": int(cluster) + 1,
            "zone_rank_by_physical_index": rank_map[int(cluster)],
            "basin_count": int(len(group)),
            "mean_physical_index": float(group["index"].mean()),
            "median_physical_index": float(group["index"].median()),
            "mean_rank_percentile": float(group["rank_pct"].mean()),
            "embedding_silhouette": sil
        })

    print(f"{name}: shape={emb.shape}; 5-zone silhouette={sil:.4f}")

# Compare independent Dink-seed zone stability (ARI invariant to label IDs)
stability = []
for a, b in [("DINK_K16_seed11", "DINK_K16_seed22"),
             ("DINK_K16_seed11", "DINK_K16_seed33"),
             ("DINK_K16_seed22", "DINK_K16_seed33")]:
    la = results[f"{a}_latent_cluster"].to_numpy()
    lb = results[f"{b}_latent_cluster"].to_numpy()
    stability.append({
        "run_a": a, "run_b": b,
        "latent_zone_ARI": float(adjusted_rand_score(la, lb))
    })
pd.DataFrame(stability).to_csv(OUT / "FSM_DINK_latent_zone_stability.csv", index=False)
pd.DataFrame(zone_summary).to_csv(OUT / "FSM_latent_zone_characteristics.csv", index=False)

# Save output tables
results.to_csv(OUT / "KRB_508_FSM_unsupervised_scores.csv", index=False)
components.to_csv(OUT / "KRB_508_FSM_physical_components.csv", index=False)

metadata = {
    "n_basins": int(len(results)),
    "n_physical_factors": int(len(FACTORS)),
    "physical_factor_directions": FACTORS,
    "physical_index_method": "Equal-weight mean of direction-corrected percentile ranks",
    "latent_zone_count": N_ZONES,
    "latent_zone_method": "StandardScaler + KMeans on saved SSL embeddings",
    "embedding_models": list(embeddings.keys()),
    "fim_used_for_training_or_scoring": False,
    "score_interpretation": "Relative susceptibility index; not a calibrated flood probability",
    "caveats": [
        "Weights are equal and are a transparent baseline, not learned optimal weights.",
        "Drainage density and some soil/land-cover effects can be context dependent.",
        "The physical index uses named mean predictors; std predictors are not included in this first baseline.",
        "Embedding zones are ranked by mean physical index; they are not flood labels.",
        "Validate against independent FIM only after freezing model choices."
    ]
}
(OUT / "FSM_method_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

print("\nOUTPUTS")
for name in [
    "KRB_508_FSM_unsupervised_scores.csv",
    "KRB_508_FSM_physical_components.csv",
    "FSM_latent_zone_characteristics.csv",
    "FSM_DINK_latent_zone_stability.csv",
    "FSM_factor_sensitivity.csv",
    "FSM_method_metadata.json"
]:
    print(" ", OUT / name)

print("\nFSM stage 2 completed.")
print("IMPORTANT: this is a transparent baseline index + SSL latent zoning,")
print("not a calibrated probability model and not yet FIM-validated.")
