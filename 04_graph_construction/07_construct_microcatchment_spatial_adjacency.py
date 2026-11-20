# ============================================================
# KOSI RIVER BASIN: 508-NODE UNDIRECTED SPATIAL GRAPH
# Based on the graph construction principle in the uploaded paper
# ============================================================

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

# ---------------- PATHS ----------------
BASIN_PATH = Path(
    r"D:\KRB\KRB-DATASET\KRB_MacroMicro"
    r"\KRB_MicroCatchments_ALL_GAPS_FILLED.shp"
)

FEATURE_PATH = Path(
    r"D:\KRB\KRB-DATASET\Harmonized_30m"
    r"\KRB_STAGE2_FEATURES_FROM_FINAL_INPUTS"
    r"\scaled_rasters\KRB_508_Basin_Feature_Matrix.csv"
)

OUT = Path(r"D:\KRB_FINAL_GRAPH_PROJECT_V6")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- CHECK INPUTS ----------------
for path in [BASIN_PATH, FEATURE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

print("Loading basin polygons...")
gdf = gpd.read_file(BASIN_PATH).reset_index(drop=True)
features = pd.read_csv(FEATURE_PATH)

N = len(gdf)

if N != 508:
    raise ValueError(f"Expected 508 basin polygons, found {N}")

if len(features) != N:
    raise ValueError(
        f"Polygon rows={N}, feature rows={len(features)}. "
        "Resolve node alignment before building model inputs."
    )

if gdf.crs is None:
    raise ValueError("Basin shapefile has no CRS.")

if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
    raise ValueError("Some basin geometries are missing or empty.")

print("Basins:", N)
print("CRS:", gdf.crs)
print("Feature rows:", len(features))

# ---------------- GEOMETRY QUALITY ----------------
invalid_count = int((~gdf.geometry.is_valid).sum())
print("Invalid geometries:", invalid_count)

if invalid_count:
    try:
        gdf.geometry = gdf.geometry.make_valid()
    except AttributeError:
        from shapely import make_valid
        gdf.geometry = gdf.geometry.apply(make_valid)

    if (~gdf.geometry.is_valid).any():
        raise ValueError(
            "Some geometries remain invalid after repair. "
            "Repair them in GIS before proceeding."
        )

# ---------------- NODE MAPPING ----------------
polygon_id_candidates = [
    "basin_id", "BASIN_ID", "Basin_ID",
    "micro_id", "MICRO_ID", "ID", "Id", "OBJECTID"
]
feature_id_candidates = [
    "basin_id", "BASIN_ID", "Basin_ID",
    "micro_id", "MICRO_ID", "ID", "Id", "OBJECTID"
]

polygon_id_col = next((c for c in polygon_id_candidates if c in gdf.columns), None)
feature_id_col = next((c for c in feature_id_candidates if c in features.columns), None)

mapping = pd.DataFrame({
    "node_index": np.arange(N, dtype=np.int64),
    "polygon_row": np.arange(N, dtype=np.int64),
})

if polygon_id_col:
    mapping["polygon_id"] = gdf[polygon_id_col].astype(str)
if feature_id_col:
    mapping["feature_id"] = features[feature_id_col].astype(str)

mapping.to_csv(OUT / "node_mapping.csv", index=False)

if polygon_id_col and feature_id_col:
    ids_match = (
        gdf[polygon_id_col].astype(str).to_numpy()
        == features[feature_id_col].astype(str).to_numpy()
    )
    print("Polygon ID column:", polygon_id_col)
    print("Feature ID column:", feature_id_col)
    print("IDs match row-by-row:", bool(ids_match.all()))
    if not ids_match.all():
        warnings.warn(
            "Basin IDs do not match row-by-row. "
            "Do not combine this graph with features until aligned."
        )
else:
    print("WARNING: matching basin ID fields were not both found.")
    print("Node-to-feature alignment remains unverified.")

# ---------------- BUILD SPATIAL ADJACENCY ----------------
print("\nBuilding polygon adjacency...")
sindex = gdf.sindex
pairs = set()
for i, geom in enumerate(gdf.geometry):
    candidates = sindex.query(geom, predicate="touches")
    for j in candidates:
        j = int(j)
        if i == j:
            continue
        a, b = sorted((i, j))
        pairs.add((a, b))
pairs = sorted(pairs)
if not pairs:
    raise RuntimeError("No touching polygon pairs found. Check geometries and CRS.")

edge_df = pd.DataFrame(pairs, columns=["source_node", "target_node"])
edge_df["weight"] = 1.0
edge_df["directed"] = False
edge_df.to_csv(OUT / "spatial_edges_undirected.csv", index=False)

edge_array = np.asarray(pairs, dtype=np.int64)
edge_index = torch.tensor(
    np.concatenate([edge_array, edge_array[:, ::-1]], axis=0).T.copy(),
    dtype=torch.long
)
edge_index = torch.unique(edge_index, dim=1)

degree = np.bincount(edge_index[0].numpy(), minlength=N)
adj = [[] for _ in range(N)]
for a, b in pairs:
    adj[a].append(b)
    adj[b].append(a)
visited = set()
component_sizes = []
for start in range(N):
    if start in visited:
        continue
    stack = [start]
    visited.add(start)
    size = 0
    while stack:
        u = stack.pop()
        size += 1
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)
    component_sizes.append(size)

graph = {
    "num_nodes": N,
    "edge_index": edge_index,
    "edge_weight": torch.ones(edge_index.shape[1]),
    "graph_type": "undirected_polygon_touch_adjacency",
    "node_order": "polygon row order; see node_mapping.csv",
    "edge_semantics": "touching basin polygons; unweighted",
}
torch.save(graph, OUT / "spatial_graph.pt")

print("\n========== SPATIAL GRAPH REPORT ==========")
print("Nodes:", N)
print("Unique undirected neighbouring pairs:", len(pairs))
print("Bidirectional edge_index entries:", edge_index.shape[1])
print("Isolated nodes:", int(np.sum(degree == 0)))
print("Minimum degree:", int(degree.min()))
print("Maximum degree:", int(degree.max()))
print("Connected components:", len(component_sizes))
print("Largest component size:", max(component_sizes))
print("Smallest component size:", min(component_sizes))
print("Self-loops:", int((edge_index[0] == edge_index[1]).sum()))
print("Unique directed edge count:", len(set(map(tuple, edge_index.T.tolist()))))
print("\nSaved files:")
print(OUT / "spatial_graph.pt")
print(OUT / "spatial_edges_undirected.csv")
print(OUT / "node_mapping.csv")
print("==========================================")
