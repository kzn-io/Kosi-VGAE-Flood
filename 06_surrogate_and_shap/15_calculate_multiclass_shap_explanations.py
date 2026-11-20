# ================================================================
# KRB VGAE-SHAP | STEP 4: MULTIMODEL SHAP EXPLAINABILITY
# Models: LightGBM, Random Forest, XGBoost
# ================================================================
# Run only after STEP 3 has completed with STATUS: PASS.
#
# Purpose:
# Explain how the three trained surrogate classifiers reproduce
# the VGAE-derived five-class target (VGAE_FSM5).
#
# IMPORTANT:
# - These are surrogate explanations, NOT causal effects.
# - These models reproduce VGAE labels; they do not directly predict
#   observed flood occurrence.
# - Original VGAE_FSM5 labels and original susceptibility maps remain
#   unchanged.
# - SHAP is calculated for all 508 basins using models fitted only on
#   the 406-row training set. The 102-row holdout remains untouched
#   during model fitting.
# ================================================================

from pathlib import Path
import json
import warnings
import traceback

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ------------------------- PATHS ----------------------------------
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "results" / "shap_analysis"
STEP1_DIR = BASE_DIR / "step01_data_qc"
STEP2_DIR = BASE_DIR / "step02_split"
STEP3_DIR = BASE_DIR / "step03_surrogate_models"
CLEAN_DATA = STEP1_DIR / "clean_modelling_data_508.csv"
STEP1_REPORT = STEP1_DIR / "qc_report.txt"
FEATURES_FILE = STEP2_DIR / "predictors.json"
STEP2_REPORT = STEP2_DIR / "split_report.txt"
STEP3_REPORT = STEP3_DIR / "model_report.txt"
MODEL_DIR = STEP3_DIR / "models"
SPATIAL_GPKG = STEP1_DIR / "aligned_microcatchments_508.gpkg"
SPATIAL_LAYER = "krb_shap_step01_basins"
OUTPUT_DIR = BASE_DIR / "step04_multimodel_shap"
PLOT_DIR = OUTPUT_DIR / "figures"
CSV_DIR = OUTPUT_DIR / "tables"
GPKG_DIR = OUTPUT_DIR / "spatial"
for folder in (OUTPUT_DIR, PLOT_DIR, CSV_DIR, GPKG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------- SETTINGS -------------------------------

TARGET = "VGAE_FSM5"
ID_COLUMNS = ["basin_id", "node_index"]
CLASS_CODES = [1, 2, 3, 4, 5]
CLASS_NAMES = {
    1: "Very_Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very_High"
}
MODEL_FILES = {
    "LightGBM": MODEL_DIR / "KRB_SHAP_LightGBM_SURROGATE.joblib",
    "Random_Forest": MODEL_DIR / "KRB_SHAP_Random_Forest_SURROGATE.joblib",
    "XGBoost": MODEL_DIR / "KRB_SHAP_XGBoost_SURROGATE.joblib"
}
RANDOM_STATE = 42
MAX_DISPLAY = 15
FIG_DPI = 600

# ------------------------- INPUT CHECKS ----------------------------

required_paths = [
    CLEAN_DATA, STEP1_REPORT, FEATURES_FILE, STEP2_REPORT,
    STEP3_REPORT, SPATIAL_GPKG, *MODEL_FILES.values()
]
missing_paths = [str(p) for p in required_paths if not p.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Required Step 1–3 input(s) missing:\n" + "\n".join(missing_paths)
    )

for report_path, step_name in [
    (STEP1_REPORT, "Step 1"),
    (STEP2_REPORT, "Step 2"),
    (STEP3_REPORT, "Step 3")
]:
    report_text = report_path.read_text(encoding="utf-8")
    if "STATUS: PASS" not in report_text:
        raise RuntimeError(f"{step_name} report is not marked STATUS: PASS.")

config = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
predictors = config["predictors"]
if len(predictors) != 34:
    raise ValueError(f"Expected 34 predictors; found {len(predictors)}.")

data = pd.read_csv(CLEAN_DATA)
required_columns = set(ID_COLUMNS + [TARGET] + predictors)
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    raise ValueError(f"Clean modelling table is missing: {sorted(missing_columns)}")

if len(data) != 508:
    raise ValueError(f"Expected 508 basins; found {len(data)}.")
if data["basin_id"].duplicated().any() or data["node_index"].duplicated().any():
    raise ValueError("Duplicate basin_id or node_index detected.")
if sorted(data[TARGET].astype(int).unique().tolist()) != CLASS_CODES:
    raise ValueError("Target must contain original class codes 1, 2, 3, 4, 5.")
if data[predictors].isna().any().any():
    raise ValueError("Predictor matrix contains missing values.")
X_all = data[predictors].astype(float)
if not np.isfinite(X_all.to_numpy()).all():
    raise ValueError("Predictor matrix contains infinite values.")

# Confirm the three models and their predictor order.
models = {}
for model_name, model_path in MODEL_FILES.items():
    model = joblib.load(model_path)
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(predictors):
        raise ValueError(
            f"{model_name} expects {model.n_features_in_} features; "
            f"the current matrix has {len(predictors)}."
        )
    models[model_name] = model

# Load the aligned 508-basin spatial layer.
try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "GeoPandas is required for spatial SHAP outputs. "
        "Install with: python -m pip install geopandas"
    ) from exc

gdf = gpd.read_file(SPATIAL_GPKG, layer=SPATIAL_LAYER)
if len(gdf) != 508:
    raise ValueError(f"Expected 508 spatial features; found {len(gdf)}.")
if not {"basin_id", "node_index", TARGET}.issubset(gdf.columns):
    raise ValueError("Spatial layer lacks basin_id, node_index or VGAE_FSM5.")

# Align geometry to the clean table using node_index and verify target labels.
gdf = gdf.drop(columns=[c for c in predictors if c in gdf.columns], errors="ignore")
check = data[["basin_id", "node_index", TARGET]].merge(
    gdf[["basin_id", "node_index", TARGET]],
    on=["basin_id", "node_index"],
    how="outer",
    suffixes=("_table", "_spatial"),
    indicator=True
)
if not (check["_merge"] == "both").all():
    raise ValueError("Spatial and tabular basin IDs do not match.")
if not (check[f"{TARGET}_table"] == check[f"{TARGET}_spatial"]).all():
    raise ValueError("Spatial and tabular VGAE_FSM5 labels do not match.")

# ------------------------- SHAP IMPORT -----------------------------

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "SHAP is not installed. Run:\n"
        "python -m pip install shap"
    ) from exc

print("SHAP version:", shap.__version__)
print("Basins:", len(X_all), "| Predictors:", len(predictors))
print("Models:", ", ".join(models.keys()))

# ------------------------- HELPERS --------------------------------

def normalise_multiclass_shap(raw_values, n_rows, n_features, n_classes):
    """
    Convert SHAP output from common SHAP versions into
    (n_rows, n_features, n_classes).
    """
    if isinstance(raw_values, list):
        # Older SHAP versions: one (rows, features) array per class.
        arrays = [np.asarray(v) for v in raw_values]
        if len(arrays) != n_classes:
            raise ValueError(
                f"SHAP returned {len(arrays)} class arrays; expected {n_classes}."
            )
        result = np.stack(arrays, axis=2)
    else:
        result = np.asarray(raw_values)

        # Newer SHAP versions generally return (rows, features, classes).
        if result.ndim == 2 and n_classes == 1:
            result = result[:, :, np.newaxis]
        elif result.ndim == 3:
            if result.shape == (n_rows, n_features, n_classes):
                pass
            elif result.shape == (n_classes, n_rows, n_features):
                result = np.transpose(result, (1, 2, 0))
            elif result.shape == (n_rows, n_classes, n_features):
                result = np.transpose(result, (0, 2, 1))
            else:
                raise ValueError(f"Unrecognised SHAP array shape: {result.shape}")
        else:
            raise ValueError(f"Unrecognised SHAP output dimensions: {result.shape}")

    if result.shape != (n_rows, n_features, n_classes):
        raise ValueError(
            f"Normalised SHAP shape {result.shape}; expected "
            f"{(n_rows, n_features, n_classes)}."
        )
    if not np.isfinite(result).all():
        raise ValueError("SHAP output contains NaN or infinite values.")
    return result


def save_bar_plot(importance_df, title, output_path, max_features=MAX_DISPLAY):
    plot_df = importance_df.head(max_features).sort_values("Mean_Abs_SHAP")
    plt.figure(figsize=(7.2, 5.4))
    plt.barh(plot_df["Feature"], plot_df["Mean_Abs_SHAP"])
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("")
    plt.title(title, loc="left", pad=9)
    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()


def save_beeswarm_plot(shap_module, shap_matrix, feature_matrix,
                       feature_names, title, output_path):
    plt.figure(figsize=(7.6, 5.8))
    shap_module.summary_plot(
        shap_matrix,
        features=feature_matrix,
        feature_names=feature_names,
        max_display=MAX_DISPLAY,
        show=False
    )
    plt.title(title, loc="left", pad=9)
    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()


# ------------------------- RUN SHAP -------------------------------

all_model_importance = []
all_class_importance = []
run_log = []
spatial_base = gdf[["basin_id", "node_index", TARGET, "geometry"]].copy()
spatial_output = GPKG_DIR / "KRB_STEP04_MULTIMODEL_SPATIAL_SHAP.gpkg"
if spatial_output.exists():
    spatial_output.unlink()

for model_name, model in models.items():
    print("\n" + "=" * 68)
    print("Calculating SHAP:", model_name)
    print("=" * 68)

    try:
        # TreeExplainer is appropriate for these tree-based classifiers.
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_all, check_additivity=False)

        # All models explain the same five original VGAE classes.
        # XGBoost was trained on encoded 0..4; output class positions
        # correspond in order to original VGAE classes 1..5.
        shap_values = normalise_multiclass_shap(
            raw_shap,
            n_rows=len(X_all),
            n_features=len(predictors),
            n_classes=5
        )

        # Long-form per-basin, per-feature, per-class attribution table.
        long_parts = []
        for class_index, class_code in enumerate(CLASS_CODES):
            class_shap = shap_values[:, :, class_index]
            # Build one row for every basin-feature pair.
            part = pd.DataFrame({
                "basin_id": np.repeat(data["basin_id"].to_numpy(), len(predictors)),
                "node_index": np.repeat(data["node_index"].to_numpy(), len(predictors)),
                "VGAE_FSM5": np.repeat(data[TARGET].astype(int).to_numpy(), len(predictors)),
                "explained_class_code": class_code,
                "explained_class_name": CLASS_NAMES[class_code],
                "feature": np.tile(np.asarray(predictors, dtype=object), len(data)),
                "feature_value": X_all.to_numpy().reshape(-1),
                "shap_value": class_shap.reshape(-1)
            })
            long_parts.append(part)

            # Class-specific global importance.
            class_imp = pd.DataFrame({
                "Feature": predictors,
                "Mean_Abs_SHAP": np.mean(np.abs(class_shap), axis=0),
                "Mean_SHAP": np.mean(class_shap, axis=0),
                "Explained_Class_Code": class_code,
                "Explained_Class_Name": CLASS_NAMES[class_code],
                "Model": model_name
            }).sort_values("Mean_Abs_SHAP", ascending=False)
            all_class_importance.append(class_imp)

            # Per-class spatial attribution: strongest local feature
            # for each basin for this explained output class.
            abs_shap = np.abs(class_shap)
            top_idx = np.argmax(abs_shap, axis=1)
            rows = np.arange(len(X_all))
            spatial_class = spatial_base.copy()
            spatial_class["explained_class_code"] = class_code
            spatial_class["explained_class_name"] = CLASS_NAMES[class_code]
            spatial_class["top_feature"] = np.asarray(predictors, dtype=object)[top_idx]
            spatial_class["top_abs_shap"] = abs_shap[rows, top_idx]
            spatial_class["top_signed_shap"] = class_shap[rows, top_idx]
            spatial_class["sum_abs_shap"] = abs_shap.sum(axis=1)
            spatial_class["mean_abs_shap"] = abs_shap.mean(axis=1)

            layer_name = f"{model_name[:8]}_class_{class_code}"
            spatial_class.to_file(
                spatial_output,
                layer=layer_name,
                driver="GPKG",
                mode="w" if not spatial_output.exists() else "a"
            )

            # Class-specific CSV table, useful for mapping individual features.
            class_csv = CSV_DIR / (
                f"KRB_SHAP_STEP04_{model_name}_CLASS_{class_code}_IMPORTANCE.csv"
            )
            class_imp.to_csv(class_csv, index=False)

        long_df = pd.concat(long_parts, ignore_index=True)
        long_csv = CSV_DIR / f"KRB_SHAP_STEP04_{model_name}_SHAP_LONG.csv"
        long_df.to_csv(long_csv, index=False)

        # Overall global importance: average absolute SHAP across all five
        # output classes, calculated separately for each feature.
        overall_imp = pd.DataFrame({
            "Feature": predictors,
            "Mean_Abs_SHAP": np.mean(np.abs(shap_values), axis=(0, 2)),
            "Mean_SHAP_All_Classes": np.mean(shap_values, axis=(0, 2)),
            "Model": model_name
        }).sort_values("Mean_Abs_SHAP", ascending=False)
        all_model_importance.append(overall_imp)

        overall_csv = CSV_DIR / f"KRB_SHAP_STEP04_{model_name}_GLOBAL_IMPORTANCE.csv"
        overall_imp.to_csv(overall_csv, index=False)

        # Overall importance plot.
        save_bar_plot(
            overall_imp,
            f"{model_name} | Global importance",
            PLOT_DIR / f"KRB_SHAP_STEP04_{model_name}_GLOBAL_BAR.png"
        )

        # Overall beeswarm: mean absolute across classes is not a signed
        # single-class explanation. Use the class-wise SHAP for each class
        # in separate beeswarm plots.
        for class_index, class_code in enumerate(CLASS_CODES):
            save_beeswarm_plot(
                shap,
                shap_values[:, :, class_index],
                X_all,
                predictors,
                f"{model_name} | {CLASS_NAMES[class_code]}",
                PLOT_DIR / (
                    f"KRB_SHAP_STEP04_{model_name}_CLASS_{class_code}_BEESWARM.png"
                )
            )

        # Save predicted labels and probabilities for all 508 basins.
        pred = model.predict(X_all).astype(int)
        if model_name == "XGBoost":
            pred = pred + 1
        pred_df = data[ID_COLUMNS + [TARGET]].copy()
        pred_df[f"{model_name}_surrogate_prediction"] = pred
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_all)
            for class_index, class_code in enumerate(CLASS_CODES):
                pred_df[f"probability_class_{class_code}"] = probabilities[:, class_index]
        pred_df.to_csv(
            CSV_DIR / f"KRB_SHAP_STEP04_{model_name}_ALL_BASIN_PREDICTIONS.csv",
            index=False
        )

        run_log.append({
            "Model": model_name,
            "Status": "PASS",
            "SHAP_Shape": str(shap_values.shape),
            "Basins_Explained": len(X_all),
            "Predictors": len(predictors),
            "Long_Table_Rows": len(long_df)
        })
        print("Completed:", model_name, "| SHAP shape:", shap_values.shape)

    except Exception as exc:
        run_log.append({
            "Model": model_name,
            "Status": "FAIL",
            "Error": repr(exc)
        })
        print(f"ERROR in {model_name}: {exc}")
        traceback.print_exc()

# ------------------------- COMPARISON TABLES ----------------------

if all_model_importance:
    global_comparison = pd.concat(all_model_importance, ignore_index=True)
    global_comparison.to_csv(
        CSV_DIR / "KRB_SHAP_STEP04_GLOBAL_IMPORTANCE_ALL_MODELS.csv",
        index=False
    )

    rank_tables = []
    for model_name in global_comparison["Model"].unique():
        temp = global_comparison[global_comparison["Model"] == model_name].copy()
        temp["Rank"] = np.arange(1, len(temp) + 1)
        rank_tables.append(temp[["Feature", "Model", "Mean_Abs_SHAP", "Rank"]])

    ranks = pd.concat(rank_tables, ignore_index=True)
    rank_pivot = ranks.pivot(
        index="Feature", columns="Model", values="Rank"
    ).reset_index()
    rank_pivot.to_csv(
        CSV_DIR / "KRB_SHAP_STEP04_CROSS_MODEL_RANKS.csv",
        index=False
    )

if all_class_importance:
    class_comparison = pd.concat(all_class_importance, ignore_index=True)
    class_comparison.to_csv(
        CSV_DIR / "KRB_SHAP_STEP04_CLASS_IMPORTANCE_ALL_MODELS.csv",
        index=False
    )

# ------------------------- FINAL REPORT ---------------------------

log_df = pd.DataFrame(run_log)
log_df.to_csv(OUTPUT_DIR / "KRB_SHAP_STEP04_RUN_STATUS.csv", index=False)

passed = int((log_df["Status"] == "PASS").sum()) if not log_df.empty else 0
failed = int((log_df["Status"] == "FAIL").sum()) if not log_df.empty else 0

report_lines = [
    "KRB VGAE-SHAP | STEP 4 MULTIMODEL SHAP REPORT",
    "=" * 72,
    f"Basins explained: {len(data)}",
    f"Environmental predictors: {len(predictors)}",
    "Models: LightGBM, Random Forest, XGBoost",
    "Explanation method: SHAP TreeExplainer",
    "Target being explained: VGAE-derived five-class label (VGAE_FSM5)",
    "Class codes: 1 Very Low; 2 Low; 3 Moderate; 4 High; 5 Very High",
    "SHAP calculation: all 508 basins; fitted models were trained on 406 training basins",
    "Spatial output: per-model, per-output-class dominant local feature attribution",
    "",
    f"Models passed: {passed}",
    f"Models failed: {failed}",
    "",
    log_df.to_string(index=False) if not log_df.empty else "No model results.",
    "",
    "INTERPRETATION LIMITS:",
    "SHAP explains each surrogate model's approximation of VGAE_FSM5.",
    "SHAP values are model attributions, not causal effects.",
    "These are not direct observed flood-occurrence predictions.",
    "Spatial attributions are conditional on the fitted surrogate models.",
    "",
    "OUTPUT FOLDER:",
    str(OUTPUT_DIR),
    "",
    f"STATUS: {'PASS' if passed == 3 and failed == 0 else 'PARTIAL' if passed > 0 else 'FAIL'}"
]
(OUTPUT_DIR / "KRB_SHAP_STEP04_REPORT.txt").write_text(
    "\n".join(report_lines), encoding="utf-8"
)

print("\n" + "=" * 72)
print("STEP 4 FINISHED")
print("=" * 72)
print(log_df.to_string(index=False))
print("Output folder:", OUTPUT_DIR)
print("Review STEP04_REPORT.txt and RUN_STATUS.csv before proceeding.")
