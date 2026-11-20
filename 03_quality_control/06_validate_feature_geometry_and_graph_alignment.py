from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(r"D:\KRB_FINAL_GRAPH_PROJECT_V6")
FEATURE_CSV = Path(r"D:\KRB\KRB-DATASET\Harmonized_30m\KRB_STAGE2_FEATURES_FROM_FINAL_INPUTS\\scaled_rasters\KRB_508_Basin_Feature_Matrix.csv")
NODE_MAP = ROOT / "node_mapping.csv"
EMBEDDING = ROOT / "VGAE_BASELINE" / "V6_VGAE_embeddings.npy"
POLYGON_GPKG = Path(r"D:\KRB\KRB-DATASET\Harmonized_30m\KRB_STAGE2_FEATURES_FROM_FINAL_INPUTS\\scaled_rasters\KRB_508_Basin_Features.gpkg")
LAYER = "basin_features"

def main():
    for p in [FEATURE_CSV, NODE_MAP, EMBEDDING, POLYGON_GPKG]:
        if not p.exists(): raise FileNotFoundError(p)
    f = pd.read_csv(FEATURE_CSV)
    m = pd.read_csv(NODE_MAP).sort_values("node_index").reset_index(drop=True)
    assert len(f) == len(m) == 508, (len(f), len(m))
    assert m.node_index.is_unique and m.feature_id.is_unique
    assert f.basin_id.is_unique
    j = m.merge(f, left_on="feature_id", right_on="basin_id", how="left", validate="one_to_one", indicator=True)
    assert (j._merge == "both").all(), j.loc[j._merge != "both", "feature_id"].tolist()[:10]
    predictors = j.drop(columns=["_merge", "basin_id"], errors="ignore").select_dtypes(include=[np.number])
    bad = predictors.columns[predictors.isna().any() | ~np.isfinite(predictors).all()].tolist()
    assert not bad, f"Missing/non-finite numeric columns: {bad}"
    emb = np.load(EMBEDDING)
    assert emb.shape == (508, 16), emb.shape
    assert np.isfinite(emb).all()
    poly = gpd.read_file(POLYGON_GPKG, layer=LAYER)
    assert len(poly) == 508, len(poly)
    key = "feature_id" if "feature_id" in poly.columns else "basin_id"
    assert poly[key].astype(str).is_unique
    print("QC PASS: 508 features, 508 mapped nodes, 508 polygons, embeddings (508,16); all numeric predictors finite.")

if __name__ == "__main__": main()
