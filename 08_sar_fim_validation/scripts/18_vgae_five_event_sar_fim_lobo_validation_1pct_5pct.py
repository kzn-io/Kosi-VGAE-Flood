# =============================================================================
# KOSI RIVER BASIN — VGAE-ONLY FIVE-EVENT SAR/FIM CLUSTER VALIDATION
# Events: 2020, 2021, 2022, 2023, 2024 (2022 added to the original script)
# Reproduces the method in:
#   KRB_FIM2020_CLUSTER_BASED_VALIDATION.py
#   KRB_FIM2020_CLUSTER_AUC_ROC_F1.py
#
# CORE METHOD (preserved):
# 1. Derive observed flood fraction for every microcatchment from valid FIM pixels.
# 2. Retain the original VGAE_FSM_cluster10 assignments (K=10).
# 3. For each basin, calculate a leave-one-basin-out (LOBO) score:
#       sum(flood pixels in OTHER same-cluster basins)
#       ------------------------------------------------
#       sum(valid FIM pixels in OTHER same-cluster basins)
# 4. Evaluate this score against exactly two basin-level targets:
#       A. Flood_fraction_ge_1pct: observed flood fraction >= 0.01.
#       B. Flood_fraction_ge_5pct: observed flood fraction >= 0.05.
# 5. ROC-AUC and average precision use the continuous LOBO score.
# 6. Threshold-dependent metrics use the median LOBO score, exactly as in
#    the original cluster AUC script. This is descriptive, not calibrated.
#
# VGAE ONLY. No DinkNet, no regionalization, no direct FSM-class AUC,
# no PSI AUC, no retraining, and no reclassification.
#
# IMPORTANT:
# - FIM coding must be 1=flood, 0=non-flood.
# - Explicit raster NoData is excluded. Other values are excluded and reported.
# - A zero-valued pixel is counted as observed non-flood only if it is not masked
#   as NoData and the user's FIM production workflow defines zero that way.
# - This is retrospective peer-cluster association, NOT independent prospective
#   flood prediction. Spatial autocorrelation and event-specificity remain.
# =============================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef
)

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------- PATHS ---------------------------------------
GPKG = Path(
    r"D:\KRB_FINAL_GRAPH_PROJECT_V6\FSM_UNSUPERVISED"
    r"\KRB_FINAL_VGAE_DINKNET_FSM_REGIONALIZATION_LAYERS"
    r"\VGAE_FSM_5CLASS.gpkg"
)
GPKG_LAYER = "vgae_fsm_5class"

NODE_FIELD = "node_index"
CLUSTER_FIELD = "VGAE_FSM_cluster10"
CLASS_FIELD = "VGAE_FSM5"  # used only for descriptive class summaries

EVENT_RASTERS = {
    "2020": Path(r"D:\KRB\SAR\FIM\2020\FIM_2020.tif"),
    "2021": Path(r"D:\KRB\SAR\FIM\2021\FIM_2021.tif"),
    "2022": Path(r"D:\KRB\SAR\FIM\2022\FIM_2022.tif"),
    "2023": Path(r"D:\KRB\SAR\FIM\2023\FIM_2023.tif"),
    "2024": Path(r"D:\KRB\SAR\FIM\2024\FIM_2024.tif"),
}

OUTPUT_DIR = Path(
    r"D:\KRB_FINAL_GRAPH_PROJECT_V6\FSM_UNSUPERVISED"
    r"\VGAE_ONLY_FIVE_EVENT_LOBO_VALIDATION"
)

FLOOD_VALUE = 1
NON_FLOOD_VALUE = 0
FLOOD_FRACTION_THRESHOLDS = {"1pct": 0.01, "5pct": 0.05}
EXPECTED_BASINS = 508
EXPECTED_CLUSTERS = 10


# ----------------------------- SMALL HELPERS ---------------------------------
def safe_div(a, b):
    return float(a / b) if b else np.nan


def metric_bundle(y_true, y_score, threshold):
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(np.uint8)

    tn, fp, fn, tp = [
        int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ]

    return {
        "Median_score_threshold": float(threshold),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": safe_div(tn, tn + fp),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Predicted_positive_basins": int(y_pred.sum()),
    }


def read_and_validate_vgae_layer():
    if not GPKG.is_file():
        raise FileNotFoundError(f"VGAE GeoPackage not found:\n{GPKG}")

    gdf = gpd.read_file(GPKG, layer=GPKG_LAYER)
    required = [NODE_FIELD, CLUSTER_FIELD, CLASS_FIELD, "geometry"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(
            f"Layer '{GPKG_LAYER}' is missing {missing}. "
            f"Available fields: {list(gdf.columns)}"
        )
    if gdf.crs is None:
        raise ValueError("VGAE polygon layer has no CRS.")
    if gdf[NODE_FIELD].duplicated().any():
        raise ValueError("Duplicate node_index values found.")
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any():
        raise ValueError("Null or empty microcatchment geometries found.")

    gdf[NODE_FIELD] = pd.to_numeric(gdf[NODE_FIELD], errors="raise").astype(int)
    gdf[CLUSTER_FIELD] = pd.to_numeric(
        gdf[CLUSTER_FIELD], errors="raise"
    ).astype(int)
    gdf[CLASS_FIELD] = pd.to_numeric(
        gdf[CLASS_FIELD], errors="raise"
    ).astype(int)

    if len(gdf) != EXPECTED_BASINS:
        raise ValueError(
            f"Expected {EXPECTED_BASINS} microcatchments; found {len(gdf)}. "
            "Check that the final 508-node VGAE layer is being used."
        )
    if gdf[NODE_FIELD].nunique() != EXPECTED_BASINS:
        raise ValueError("node_index does not contain 508 unique values.")
    if gdf[CLUSTER_FIELD].nunique() != EXPECTED_CLUSTERS:
        raise ValueError(
            f"Expected {EXPECTED_CLUSTERS} VGAE clusters; "
            f"found {gdf[CLUSTER_FIELD].nunique()}."
        )
    if not set(gdf[CLASS_FIELD].unique()).issubset({1, 2, 3, 4, 5}):
        raise ValueError("VGAE_FSM5 contains values outside classes 1–5.")

    return gdf.sort_values(NODE_FIELD).reset_index(drop=True)


# -------------------------- EVENT FIM EXTRACTION -----------------------------
def extract_event_basin_counts(gdf, event_name, raster_path):
    if not raster_path.is_file():
        raise FileNotFoundError(
            f"FIM raster for {event_name} not found:\n{raster_path}"
        )

    records = []
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"{event_name}: FIM raster has no CRS.")

        print(f"\n[{event_name}] Raster CRS: {src.crs}")
        print(f"[{event_name}] Raster size: {src.width} x {src.height}")
        print(f"[{event_name}] Resolution: {src.res}")
        print(f"[{event_name}] NoData: {src.nodata}")
        print(f"[{event_name}] Bounds: {src.bounds}")

        if src.nodata == NON_FLOOD_VALUE:
            raise ValueError(
                f"{event_name}: raster NoData is 0, but 0 is also the "
                "non-flood class. Correct the FIM encoding/NoData definition "
                "before validation; otherwise valid non-flood pixels cannot "
                "be distinguished from NoData."
            )

        projected = gdf.to_crs(src.crs)

        for _, basin in projected.iterrows():
            geom = basin.geometry
            base = {
                "Event": event_name,
                "node_index": int(basin[NODE_FIELD]),
                "Cluster_ID": int(basin[CLUSTER_FIELD]),
                "FSM_class": int(basin[CLASS_FIELD]),
            }

            try:
                cropped, _ = mask(
                    src,
                    [geom],
                    crop=True,
                    filled=False,
                    all_touched=False
                )
                values = cropped[0].compressed()
            except ValueError:
                values = np.array([], dtype=np.float64)

            # Keep only explicitly defined binary FIM observations.
            flood_n = int(np.count_nonzero(values == FLOOD_VALUE))
            nonflood_n = int(np.count_nonzero(values == NON_FLOOD_VALUE))
            excluded_n = int(len(values) - flood_n - nonflood_n)
            valid_n = flood_n + nonflood_n

            frac = flood_n / valid_n if valid_n else np.nan

            base.update({
                "FIM_pixel_count": valid_n,
                "Observed_flood_pixels": flood_n,
                "Observed_nonflood_pixels": nonflood_n,
                "Excluded_nonbinary_pixels": excluded_n,
                "Flood_fraction": frac,
                "Flood_fraction_percent": 100.0 * frac if valid_n else np.nan,
                "Target_flood_fraction_ge_1pct": (
                    int(frac >= FLOOD_FRACTION_THRESHOLDS["1pct"]) if valid_n else np.nan
                ),
                "Target_flood_fraction_ge_5pct": (
                    int(frac >= FLOOD_FRACTION_THRESHOLDS["5pct"]) if valid_n else np.nan
                ),
            })
            records.append(base)

    basin_df = pd.DataFrame(records).sort_values("node_index").reset_index(drop=True)

    if len(basin_df) != EXPECTED_BASINS:
        raise RuntimeError(f"{event_name}: expected 508 basin rows, got {len(basin_df)}.")
    if basin_df["node_index"].duplicated().any():
        raise RuntimeError(f"{event_name}: duplicate node_index after extraction.")

    return basin_df


# ---------------------------- ORIGINAL LOBO METHOD ---------------------------
def add_lobo_peer_cluster_scores(basin_df):
    """
    Faithful implementation of original KRB_FIM2020_CLUSTER_AUC_ROC_F1.py:
    pixel-weighted flood fraction of OTHER valid basins in the same cluster.
    """
    out = basin_df.copy()
    scores = np.full(len(out), np.nan, dtype=float)

    cluster_ids = out["Cluster_ID"].to_numpy()
    flood_counts = out["Observed_flood_pixels"].to_numpy(dtype=float)
    valid_counts = out["FIM_pixel_count"].to_numpy(dtype=float)

    for i in range(len(out)):
        peers = (
            (cluster_ids == cluster_ids[i]) &
            (np.arange(len(out)) != i) &
            (valid_counts > 0)
        )
        denominator = valid_counts[peers].sum()
        if denominator > 0:
            scores[i] = flood_counts[peers].sum() / denominator

    out["LOBO_cluster_flood_score"] = scores
    return out


def evaluate_event_targets(basin_df, event_name):
    scored = basin_df.loc[
        basin_df["LOBO_cluster_flood_score"].notna() &
        (basin_df["FIM_pixel_count"] > 0)
    ].copy()

    metric_rows = []
    roc_rows = []

    for target_col, target_label in [
        ("Target_flood_fraction_ge_1pct", "Observed flood fraction >= 1%"),
        ("Target_flood_fraction_ge_5pct", "Observed flood fraction >= 5%"),
    ]:
        target_data = scored.dropna(subset=[target_col]).copy()
        y = target_data[target_col].astype(int).to_numpy()
        score = target_data["LOBO_cluster_flood_score"].astype(float).to_numpy()

        n = len(y)
        positives = int(y.sum())
        negatives = int(n - positives)
        median_score = float(np.median(score)) if n else np.nan

        row = {
            "Event": event_name,
            "Target": target_col,
            "Target_definition": target_label,
            "Evaluation_unit": "microcatchment",
            "Score_definition": (
                "LOBO pixel-weighted observed flood fraction of other "
                "microcatchments in the same original VGAE K=10 cluster"
            ),
            "N_basins": n,
            "Positive_basins": positives,
            "Negative_basins": negatives,
            "Positive_prevalence": safe_div(positives, n),
            "ROC_AUC": np.nan,
            "Average_precision_PR_AUC": np.nan,
        }

        if n and len(np.unique(y)) == 2:
            row["ROC_AUC"] = float(roc_auc_score(y, score))
            row["Average_precision_PR_AUC"] = float(
                average_precision_score(y, score)
            )
            row.update(metric_bundle(y, score, median_score))

            fpr, tpr, thresholds = roc_curve(y, score)
            for a, b, c in zip(fpr, tpr, thresholds):
                roc_rows.append({
                    "Event": event_name,
                    "Target": target_col,
                    "False_positive_rate": float(a),
                    "True_positive_rate": float(b),
                    "Threshold": float(c),
                })
        elif n:
            row.update(metric_bundle(y, score, median_score))
            print(
                f"WARNING {event_name} {target_col}: only one target class; "
                "ROC-AUC and AP are undefined."
            )

        metric_rows.append(row)

    return pd.DataFrame(metric_rows), pd.DataFrame(roc_rows), scored


# --------------------------- DESCRIPTIVE TABLES ------------------------------
def cluster_flood_summary(basin_df):
    rows = []
    for cluster_id, group in basin_df.groupby("Cluster_ID", sort=True):
        valid_pixels = int(group["FIM_pixel_count"].sum())
        flood_pixels = int(group["Observed_flood_pixels"].sum())
        rows.append({
            "Event": group["Event"].iloc[0],
            "Cluster_ID": int(cluster_id),
            "N_basins": int(len(group)),
            "Basins_with_valid_FIM": int((group["FIM_pixel_count"] > 0).sum()),
            "FIM_pixel_count": valid_pixels,
            "Observed_flood_pixels": flood_pixels,
            "Observed_nonflood_pixels": int(
                group["Observed_nonflood_pixels"].sum()
            ),
            "Pixel_weighted_flood_fraction": safe_div(flood_pixels, valid_pixels),
            "Pixel_weighted_flood_fraction_percent": (
                100.0 * safe_div(flood_pixels, valid_pixels)
                if valid_pixels else np.nan
            ),
            "Mean_basin_flood_fraction": group["Flood_fraction"].mean(),
            "Median_basin_flood_fraction": group["Flood_fraction"].median(),
        })
    return pd.DataFrame(rows)


def class_flood_summary(basin_df):
    rows = []
    for class_id, group in basin_df.groupby("FSM_class", sort=True):
        valid_pixels = int(group["FIM_pixel_count"].sum())
        flood_pixels = int(group["Observed_flood_pixels"].sum())
        rows.append({
            "Event": group["Event"].iloc[0],
            "FSM_class": int(class_id),
            "N_basins": int(len(group)),
            "FIM_pixel_count": valid_pixels,
            "Observed_flood_pixels": flood_pixels,
            "Pixel_weighted_flood_fraction": safe_div(flood_pixels, valid_pixels),
            "Pixel_weighted_flood_fraction_percent": (
                100.0 * safe_div(flood_pixels, valid_pixels)
                if valid_pixels else np.nan
            ),
            "Mean_basin_flood_fraction": group["Flood_fraction"].mean(),
            "Median_basin_flood_fraction": group["Flood_fraction"].median(),
            "At_least_1pct_rate": group["Target_flood_fraction_ge_1pct"].mean(),
            "At_least_5pct_rate": group["Target_flood_fraction_ge_5pct"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf = read_and_validate_vgae_layer()

    print("=" * 78)
    print("KRB — VGAE-ONLY FIVE-EVENT LOBO CLUSTER VALIDATION")
    print("=" * 78)
    print("GeoPackage:", GPKG)
    print("Layer:", GPKG_LAYER)
    print("Basins:", len(gdf))
    print("VGAE clusters:", sorted(gdf[CLUSTER_FIELD].unique().tolist()))
    print("Five-class counts:", gdf[CLASS_FIELD].value_counts().sort_index().to_dict())
    print("NOTE: cluster IDs and FSM classes are read-only; no model changes occur.")

    all_basin = []
    all_metrics = []
    all_roc = []
    all_cluster = []
    all_class = []
    qc_rows = []

    for event_name, raster_path in EVENT_RASTERS.items():
        print("\n" + "#" * 78)
        print("PROCESSING EVENT:", event_name)
        print("#" * 78)

        basin_df = extract_event_basin_counts(gdf, event_name, raster_path)
        basin_df = add_lobo_peer_cluster_scores(basin_df)

        metric_df, roc_df, scored = evaluate_event_targets(basin_df, event_name)
        cluster_df = cluster_flood_summary(basin_df)
        class_df = class_flood_summary(basin_df)

        all_basin.append(basin_df)
        all_metrics.append(metric_df)
        all_roc.append(roc_df)
        all_cluster.append(cluster_df)
        all_class.append(class_df)

        valid_basins = int((basin_df["FIM_pixel_count"] > 0).sum())
        flood_pixels = int(basin_df["Observed_flood_pixels"].sum())
        valid_pixels = int(basin_df["FIM_pixel_count"].sum())
        excluded_pixels = int(basin_df["Excluded_nonbinary_pixels"].sum())

        qc_rows.append({
            "Event": event_name,
            "Basins_total": int(len(basin_df)),
            "Basins_with_valid_FIM": valid_basins,
            "Basins_without_valid_FIM": int(len(basin_df) - valid_basins),
            "Total_valid_FIM_pixels": valid_pixels,
            "Total_flood_pixels": flood_pixels,
            "Total_nonflood_pixels": int(
                basin_df["Observed_nonflood_pixels"].sum()
            ),
            "Excluded_nonbinary_pixels": excluded_pixels,
            "Observed_flood_pixel_prevalence": safe_div(flood_pixels, valid_pixels),
            "Basins_ge_1pct": int(
                basin_df["Target_flood_fraction_ge_1pct"].fillna(0).sum()
            ),
            "Basins_ge_5pct": int(
                basin_df["Target_flood_fraction_ge_5pct"].fillna(0).sum()
            ),
            "Basins_with_LOBO_score": int(
                basin_df["LOBO_cluster_flood_score"].notna().sum()
            ),
        })

        print("\nEvent QC")
        print("Valid basins:", valid_basins, "/", len(basin_df))
        print("Valid FIM pixels:", f"{valid_pixels:,}")
        print("Flood pixels:", f"{flood_pixels:,}")
        print("Excluded nonbinary pixels:", f"{excluded_pixels:,}")
        print("\nLOBO validation metrics")
        print(metric_df.to_string(index=False))
        print("\nCluster flood fractions")
        print(cluster_df[[
            "Cluster_ID", "N_basins", "FIM_pixel_count",
            "Observed_flood_pixels", "Pixel_weighted_flood_fraction_percent"
        ]].to_string(index=False))

    basin_all = pd.concat(all_basin, ignore_index=True)
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    roc_all = pd.concat(all_roc, ignore_index=True)
    cluster_all = pd.concat(all_cluster, ignore_index=True)
    class_all = pd.concat(all_class, ignore_index=True)
    qc_df = pd.DataFrame(qc_rows)

    # Save CSV outputs.
    basin_all.to_csv(OUTPUT_DIR / "VGAE_LOBO_Basin_Event_Scores.csv", index=False)
    metrics_all.to_csv(OUTPUT_DIR / "VGAE_LOBO_MultiEvent_Metrics_1PCT_5PCT.csv", index=False)
    roc_all.to_csv(OUTPUT_DIR / "VGAE_LOBO_ROC_Curve_Points.csv", index=False)
    cluster_all.to_csv(OUTPUT_DIR / "VGAE_Cluster_Flood_Fractions.csv", index=False)
    class_all.to_csv(OUTPUT_DIR / "VGAE_FSM5_Descriptive_Flood_Summary.csv", index=False)
    qc_df.to_csv(OUTPUT_DIR / "VGAE_MultiEvent_FIM_QC.csv", index=False)

    # Save one consolidated workbook.
    xlsx_path = OUTPUT_DIR / "VGAE_ONLY_FIVE_EVENT_LOBO_VALIDATION_1PCT_5PCT.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        metrics_all.to_excel(writer, sheet_name="LOBO_Metrics", index=False)
        qc_df.to_excel(writer, sheet_name="Event_QC", index=False)
        basin_all.to_excel(writer, sheet_name="Basin_Event_Scores", index=False)
        cluster_all.to_excel(writer, sheet_name="Cluster_Flood_Fractions", index=False)
        class_all.to_excel(writer, sheet_name="FSM5_Descriptive", index=False)
        roc_all.to_excel(writer, sheet_name="ROC_Curve_Points", index=False)

    # Record reproducibility metadata and explicit method.
    metadata = {
        "project": "Kosi River Basin VGAE-only five-event SAR/FIM validation",
        "gpkg": str(GPKG),
        "gpkg_layer": GPKG_LAYER,
        "node_field": NODE_FIELD,
        "cluster_field": CLUSTER_FIELD,
        "class_field_descriptive_only": CLASS_FIELD,
        "expected_basins": EXPECTED_BASINS,
        "expected_clusters": EXPECTED_CLUSTERS,
        "events": {k: str(v) for k, v in EVENT_RASTERS.items()},
        "fim_coding": {"flood": FLOOD_VALUE, "nonflood": NON_FLOOD_VALUE},
        "flood_fraction_target_thresholds": FLOOD_FRACTION_THRESHOLDS,
        "score": (
            "For each basin, pixel-weighted flood fraction of all other "
            "valid-FIM basins in the same original VGAE cluster."
        ),
        "binary_metric_threshold": (
            "Median LOBO score within each event and target; descriptive only."
        ),
        "excluded_methods": [
            "DinkNet", "regionalization", "direct FSM-class ROC-AUC",
            "PSI ROC-AUC", "retraining", "reclassification"
        ],
        "interpretation": (
            "Retrospective spatial association, not prospective independent "
            "prediction. Cluster-derived scores share spatial structure and "
            "are event-specific."
        )
    }
    with open(OUTPUT_DIR / "VGAE_VALIDATION_METHOD_METADATA.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 78)
    print("VALIDATION FINISHED")
    print("Output directory:", OUTPUT_DIR)
    print("Excel:", xlsx_path)
    print("\nFinal event-wise metrics:")
    print(metrics_all[[
        "Event", "Target", "N_basins", "Positive_basins",
        "Negative_basins", "Positive_prevalence",
        "ROC_AUC", "Average_precision_PR_AUC",
        "Balanced_accuracy", "Precision", "Recall_sensitivity", "F1", "MCC"
    ]].to_string(index=False))
    print("""
INTERPRETATION NOTES:
- The LOBO score reproduces the original VGAE cluster validation logic.
- FSM five-class values are used only for descriptive class-wise summaries.
- The original K=10 cluster assignments are not modified.
- AP must be interpreted against the positive prevalence for each target/event.
- The median-score threshold is descriptive, not an optimized or calibrated
  susceptibility threshold.
- These results are retrospective peer-cluster association, not independent
  prospective flood prediction.
- Verify the FIM source workflow, binary coding, permanent-water treatment,
  raster coverage and event dates before publication.
""")


if __name__ == "__main__":
    main()
