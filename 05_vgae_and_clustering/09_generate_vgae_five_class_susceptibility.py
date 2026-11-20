# KRB FINAL FSM — VGAE ONLY; Dink-Net and regionalization excluded.
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")

ROOT = Path(r"D:\KRB_FINAL_GRAPH_PROJECT_V6")
DATA = Path(r"D:\KRB\KRB-DATASET\Harmonized_30m\KRB_STAGE2_FEATURES_FROM_FINAL_INPUTS\scaled_rasters")
OUT = ROOT / "FSM_UNSUPERVISED" / "FINAL_VGAE_FSM_ONLY"
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_CSV = DATA / "KRB_508_Basin_Feature_Matrix.csv"
NODE_MAP = ROOT / "node_mapping.csv"
EMBEDDING = ROOT / "VGAE_BASELINE" / "V6_VGAE_embeddings.npy"
SOURCE_GPKG = DATA / "KRB_508_Basin_Features.gpkg"
SOURCE_LAYER = "basin_features"
K = 10
SEED = 11
N_INIT = 50
DIRECTIONS = {
    "DEM_scaled_mean": -1,
    "Slope_scaled_mean": -1,
    "Distance_to_Main_River_scaled_mean": -1,
    "Drainage_Density_scaled_mean": 1,
    "Max1Day_Rainfall_scaled_mean": 1,
    "Mean_Rainfall_scaled_mean": 1,
    "TWI_scaled_mean": 1,
}
NAMES = {1:"Very Low", 2:"Low", 3:"Moderate", 4:"High", 5:"Very High"}

def main():
    for p in (FEATURE_CSV, NODE_MAP, EMBEDDING, SOURCE_GPKG):
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")
    features = pd.read_csv(FEATURE_CSV)
    mapping = pd.read_csv(NODE_MAP).sort_values("node_index").reset_index(drop=True)
    nodes = mapping.merge(features, left_on="feature_id", right_on="basin_id",
                          how="left", validate="one_to_one", indicator=True)
    if (nodes["_merge"] != "both").any():
        raise ValueError("Feature ID join failed; inspect feature_id and basin_id.")
    nodes = nodes.drop(columns="_merge")
    if len(nodes) != 508 or nodes.node_index.nunique() != 508:
        raise ValueError(f"Expected 508 unique nodes; got {len(nodes)}.")
    missing = [c for c in DIRECTIONS if c not in nodes.columns]
    if missing:
        raise ValueError(f"Missing physical-index fields: {missing}")
    adjusted = {}
    for col, direction in DIRECTIONS.items():
        r = pd.to_numeric(nodes[col], errors="coerce").rank(pct=True, method="average")
        adjusted[col] = r if direction == 1 else 1-r
    components = pd.DataFrame(adjusted)
    nodes["physical_susceptibility_index"] = components.mean(axis=1)
    emb = np.load(EMBEDDING)
    if emb.ndim != 2 or emb.shape[0] != 508:
        raise ValueError(f"Unexpected VGAE embedding shape: {emb.shape}")
    nodes["VGAE_cluster_K10"] = KMeans(n_clusters=K, n_init=N_INIT,
                                        random_state=SEED).fit_predict(emb) + 1
    stats = nodes.groupby("VGAE_cluster_K10").agg(
        basin_count=("feature_id","size"),
        mean_physical_index=("physical_susceptibility_index","mean"),
        median_physical_index=("physical_susceptibility_index","median")
    ).reset_index().sort_values(
        ["median_physical_index","mean_physical_index"]
    ).reset_index(drop=True)
    stats["susceptibility_rank"] = np.arange(1, len(stats)+1)
    stats["FSM_5class"] = ((stats.susceptibility_rank-1)//2+1).astype(int)
    stats["FSM_5class_name"] = stats.FSM_5class.map(NAMES)
    class_map = stats.set_index("VGAE_cluster_K10").FSM_5class
    nodes["FSM_5class"] = nodes.VGAE_cluster_K10.map(class_map).astype(int)
    nodes["FSM_5class_name"] = nodes.FSM_5class.map(NAMES)
    for col in components:
        nodes[col+"_adjusted_percentile"] = components[col]

    polygons = gpd.read_file(SOURCE_GPKG, layer=SOURCE_LAYER)
    key = "feature_id" if "feature_id" in polygons.columns else (
        "basin_id" if "basin_id" in polygons.columns else None)
    if key is None:
        raise ValueError("Polygon layer requires feature_id or basin_id.")
    polygons["__key"] = polygons[key].astype(str)
    attrs = nodes[["feature_id","node_index","VGAE_cluster_K10","FSM_5class",
                   "FSM_5class_name","physical_susceptibility_index"]].copy()
    attrs["__key"] = attrs.feature_id.astype(str)
    if polygons.__key.duplicated().any() or attrs.__key.duplicated().any():
        raise ValueError("Duplicate join keys found; stopping to prevent incorrect joins.")
    result = polygons.merge(attrs.drop(columns="feature_id"), on="__key",
                            how="left", validate="one_to_one")
    result = gpd.GeoDataFrame(result.drop(columns="__key"),
                              geometry=polygons.geometry.name, crs=polygons.crs)
    if result.node_index.isna().any():
        raise ValueError(f"Unmatched polygons: {result.node_index.isna().sum()}")
    gpkg = OUT / "KRB_VGAE_FSM_5CLASS.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    result.to_file(gpkg, layer="vgae_fsm_5class", driver="GPKG")
    nodes.to_csv(OUT/"KRB_508_VGAE_FSM_node_assignments.csv", index=False)
    stats.to_csv(OUT/"VGAE_FSM_cluster_to_5class_lookup.csv", index=False)
    counts = nodes.FSM_5class.value_counts().reindex([1,2,3,4,5], fill_value=0)
    count_df = pd.DataFrame({"FSM_5class":counts.index, "class_name":[NAMES[i] for i in counts.index],
                             "basin_count":counts.values, "percent":counts.values/508*100})
    count_df.to_csv(OUT/"VGAE_FSM_5class_counts.csv", index=False)
    metadata = {
        "scope":"VGAE-based five-class relative FSM only",
        "excluded":["Dink-Net","regionalization"],
        "nodes":508, "KMeans_K":K, "seed":SEED, "n_init":N_INIT,
        "physical_index_factors":DIRECTIONS,
        "classification":"Rank the 10 VGAE clusters by median physical index; group consecutive pairs into five ordinal classes.",
        "caution":"Relative susceptibility, not calibrated flood probability. Independent FIM validation remains necessary.",
        "output":str(gpkg), "layer":"vgae_fsm_5class"
    }
    (OUT/"VGAE_FSM_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("\n=== VGAE FSM ONLY COMPLETE ===")
    print("Dink-Net: excluded | Regionalization: excluded")
    print(count_df.to_string(index=False))
    print("\nGeoPackage:",gpkg)
    print("Layer: vgae_fsm_5class")
    print("Output folder:",OUT)

if __name__ == "__main__":
    main()
