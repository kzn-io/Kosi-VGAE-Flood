# ================================================================
# KRB VGAE-SHAP | STEP 05 — FOUR STANDARD SHAP FIGURES
# Selected surrogate: LightGBM
# Target: VGAE-derived five-class susceptibility (VGAE_FSM5)
# Study units: 508 microcatchments
#
# FIGURES:
# 1. Multiclass SHAP beeswarm (one panel per class; top 10/class)
# 2. Global SHAP bar plot (macro-average mean |SHAP| across 5 classes)
# 3. Representative Very High microcatchment waterfall (class 5)
# 4. Class-5 SHAP dependence plots for the four most important features
#
# IMPORTANT:
# - Explanations concern the LightGBM surrogate's approximation of
#   VGAE-derived class labels, NOT observed flood occurrence.
# - SHAP values are model-output attributions, not causal effects.
# - This workflow uses the existing 508-row clean modelling table.
# - The predictors in that table are scaled. Dependence x-axes therefore
#   show scaled predictor values unless unscaled data are substituted.
# - Waterfall output is the model's class output, NOT flood probability.
# ================================================================

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ------------------------- PATHS ----------------------------------
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "shap_analysis"
STEP1 = BASE / "step01_data_qc"
STEP2 = BASE / "step02_split"
STEP3 = BASE / "step03_surrogate_models"
DATA_FILE = STEP1 / "clean_modelling_data_508.csv"
FEATURES_FILE = STEP2 / "predictors.json"
MODEL_FILE = STEP3 / "models" / "KRB_SHAP_LightGBM_SURROGATE.joblib"
OUT = BASE / "step05_lightgbm_shap"
FIG = OUT / "figures"
TAB = OUT / "tables"
for folder in (OUT, FIG, TAB):
    folder.mkdir(parents=True, exist_ok=True)
print(f"SHAP outputs: {OUT}")

# ------------------------- SETTINGS -------------------------------
TARGET = "VGAE_FSM5"
ID_COLS = ["basin_id", "node_index"]
CLASS_CODES = [1, 2, 3, 4, 5]
CLASS_NAMES = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}
N_GLOBAL = 10
N_DEPENDENCE = 4
RANDOM_STATE = 42

# Publication palette: class colours and a restrained low-to-high gradient.
PAPER = "#FAF7F0"
INK = "#26343A"
MUTED = "#64737A"
GRID = "#E5DED2"
CLASS_COLORS = {
    1: "#4776A8",
    2: "#287C78",
    3: "#D19A36",
    4: "#C66D4F",
    5: "#80618F",
}
FEATURE_CMAP = LinearSegmentedColormap.from_list(
    "KosiSHAP_Bright",
    ["#075BFF", "#FFF000", "#FF1744"],
    N=256
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

def feature_label(raw):
    """Complete concise labels; no 'basin mean' wording."""
    raw = str(raw)
    suffix = ""
    for ending, new_suffix in (
        ("_scaled_mean", ""),
        ("_scaled_std", " variability"),
        ("_fraction", " share"),
    ):
        if raw.endswith(ending):
            raw = raw[:-len(ending)]
            suffix = new_suffix
            break
    names = {
        "DEM": "Elevation",
        "Slope": "Terrain slope",
        "TWI": "Topographic wetness index",
        "Distance_to_Main_River": "Distance to main river",
        "Drainage_Density": "Drainage density",
        "ET": "Evapotranspiration",
        "Max1Day_Rainfall": "Maximum 1-day rainfall",
        "Mean_Rainfall": "Mean rainfall",
        "SMAP": "Surface soil moisture (SMAP)",
        "SPI": "Standardised precipitation index",
        "Clay_0_5cm": "Clay content (0–5 cm)",
        "Sand_0_5cm": "Sand content (0–5 cm)",
        "Silt_0_5cm": "Silt content (0–5 cm)",
        "WorldCover_10": "Tree cover",
        "WorldCover_20": "Shrubland",
        "WorldCover_30": "Grassland",
        "WorldCover_40": "Cropland",
        "WorldCover_50": "Built-up land",
        "WorldCover_60": "Bare / sparse vegetation",
        "WorldCover_70": "Snow and ice",
        "WorldCover_80": "Permanent water",
        "WorldCover_90": "Herbaceous wetland",
        "WorldCover_95": "Mangroves",
        "WorldCover_100": "Moss and lichen",
    }
    return names.get(raw, raw.replace("_", " ")) + suffix

def style_axis(ax, grid_axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, pad=6)

def save_figure(fig, name):
    """Save PNG and PDF in the local output folder."""
    FIG.mkdir(parents=True, exist_ok=True)
    png_path = FIG / name
    pdf_path = FIG / name.replace(".png", ".pdf")
    try:
        fig.savefig(str(png_path), dpi=600, bbox_inches="tight",
                    facecolor=PAPER, edgecolor="none")
        fig.savefig(str(pdf_path), bbox_inches="tight",
                    facecolor=PAPER, edgecolor="none")
    except OSError as exc:
        raise OSError(
            f"Could not save SHAP figure. Output folder: {FIG}\n"
            f"PNG target: {png_path}\nOriginal error: {exc}\n"
            "Check folder permissions and close any program locking the output file."
        ) from exc
    finally:
        plt.close(fig)

def as_class_arrays(raw_values, n_rows, n_features, n_classes):
    """Normalise SHAP output variants to (rows, features, classes)."""
    if isinstance(raw_values, list):
        if len(raw_values) != n_classes:
            raise ValueError(
                f"SHAP returned {len(raw_values)} class arrays; expected {n_classes}."
            )
        arr = np.stack([np.asarray(v) for v in raw_values], axis=-1)
    else:
        arr = np.asarray(raw_values)
        if arr.ndim == 2 and n_classes == 1:
            arr = arr[:, :, None]
        elif arr.ndim == 3:
            # SHAP versions have returned either (n,p,k) or (k,n,p).
            if arr.shape == (n_classes, n_rows, n_features):
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.shape != (n_rows, n_features, n_classes):
                raise ValueError(f"Unrecognised SHAP array shape: {arr.shape}")
        else:
            raise ValueError(f"Unrecognised SHAP output dimensions: {arr.shape}")
    if arr.shape != (n_rows, n_features, n_classes):
        raise ValueError(
            f"Expected SHAP shape {(n_rows,n_features,n_classes)}, got {arr.shape}"
        )
    return arr

# ------------------------- INPUT VALIDATION -----------------------
for path in (DATA_FILE, FEATURES_FILE, MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input:\n{path}\n"
            "Confirm that Steps 1–3 have completed on the project computer."
        )

cfg = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
predictors = cfg["predictors"]
data = pd.read_csv(DATA_FILE)

required = set(ID_COLS + [TARGET] + predictors)
missing = sorted(required.difference(data.columns))
if missing:
    raise ValueError(f"Clean modelling table lacks required columns: {missing}")
if len(data) != 508:
    raise ValueError(f"Expected exactly 508 microcatchments; found {len(data)}.")
if data["basin_id"].duplicated().any() or data["node_index"].duplicated().any():
    raise ValueError("Duplicate basin_id or node_index found.")
if sorted(data[TARGET].astype(int).unique().tolist()) != CLASS_CODES:
    raise ValueError("Target must contain class codes 1, 2, 3, 4 and 5.")
if data[predictors].isna().any().any():
    raise ValueError("Missing predictor values detected.")

X = data[predictors].astype(float)
if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Infinite predictor values detected.")
y = data[TARGET].astype(int).to_numpy()

model = joblib.load(MODEL_FILE)
model_classes = [int(v) for v in getattr(model, "classes_", CLASS_CODES)]
if model_classes != CLASS_CODES:
    raise ValueError(
        f"Model class order {model_classes} does not match {CLASS_CODES}. "
        "Do not proceed until the class encoding is reconciled."
    )

print(f"Loaded {len(data)} microcatchments and {len(predictors)} predictors.")
print("Loaded existing LightGBM surrogate model.")

# ------------------------- SHAP CALCULATION -----------------------
# TreeExplainer uses the already-fitted model; this script does not refit it.
explainer = shap.TreeExplainer(model)
raw_shap = explainer.shap_values(X, check_additivity=False)
shap_values = as_class_arrays(
    raw_shap, len(X), len(predictors), len(CLASS_CODES)
)

expected = np.asarray(explainer.expected_value)
if expected.ndim == 0:
    base_values = np.repeat(float(expected), len(CLASS_CODES))
else:
    base_values = expected.reshape(-1)
if len(base_values) != len(CLASS_CODES):
    raise ValueError(
        f"Expected {len(CLASS_CODES)} class base values; got {len(base_values)}."
    )

# Store long-format attribution data for reproducibility and future plotting.
records = []
for k, class_code in enumerate(CLASS_CODES):
    block = pd.DataFrame({
        "basin_id": data["basin_id"].to_numpy(),
        "node_index": data["node_index"].to_numpy(),
        "actual_VGAE_FSM5": y,
        "predicted_class": model.predict(X).astype(int),
    })
    for j, feat in enumerate(predictors):
        records.append(pd.DataFrame({
            "basin_id": block["basin_id"],
            "node_index": block["node_index"],
            "actual_VGAE_FSM5": block["actual_VGAE_FSM5"],
            "predicted_class": block["predicted_class"],
            "explained_class_code": class_code,
            "feature": feat,
            "feature_value": X[feat].to_numpy(),
            "shap_value": shap_values[:, j, k],
        }))
shap_long = pd.concat(records, ignore_index=True)
shap_long.to_csv(TAB / "LightGBM_SHAP_LONG_ALL_508.csv", index=False)

# Prediction audit: class and probability for each microcatchment.
predicted = model.predict(X).astype(int)
prob = model.predict_proba(X)
pred_audit = data[ID_COLS + [TARGET]].copy()
pred_audit["LightGBM_predicted_class"] = predicted
for idx, code in enumerate(model_classes):
    pred_audit[f"Probability_class_{code}"] = prob[:, idx]
pred_audit.to_csv(TAB / "LightGBM_Prediction_Audit.csv", index=False)

# ---------------- FIGURE 1: MULTICLASS GLOBAL BEESWARM -------------
# A multiclass model has five separate class outputs. A single beeswarm
# cannot honestly collapse these into one signed SHAP axis, so use five
# class panels, each showing its own top ten predictors and all 508 rows.
fig, axes = plt.subplots(5, 1, figsize=(14.2, 15.0))
for k, (code, ax) in enumerate(zip(CLASS_CODES, axes)):
    values = shap_values[:, :, k]
    mean_abs = np.mean(np.abs(values), axis=0)
    top_idx = np.argsort(mean_abs)[-N_GLOBAL:]
    order = top_idx[np.argsort(mean_abs[top_idx])]
    labels = [feature_label(predictors[i]) for i in order]
    exp = shap.Explanation(
        values=values[:, order],
        base_values=np.repeat(base_values[k], len(X)),
        data=X.iloc[:, order].to_numpy(),
        feature_names=labels,
    )
    plt.sca(ax)
    shap.plots.beeswarm(
        exp, max_display=N_GLOBAL, show=False, plot_size=None,
        color=FEATURE_CMAP, alpha=0.82
    )
    ax.set_title(CLASS_NAMES[code], loc="left",
                 color=CLASS_COLORS[code], fontsize=10.5, pad=5)
    ax.set_xlabel("SHAP value (LightGBM class output)")
    ax.set_xlim(left=min(-0.01, ax.get_xlim()[0]))
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
fig.suptitle("SHAP summary by class",
             x=0.075, ha="left", y=0.999, fontsize=15, fontweight="bold")
fig.text(0.075, 0.982,
         "LightGBM surrogate • 508 microcatchments • top 10 predictors per class",
         fontsize=8.5, color=MUTED)
fig.text(0.075, 0.006,
         "Red = relatively high predictor value; blue = relatively low. "
         "SHAP explains VGAE-derived labels, not observed flood occurrence.",
         fontsize=7.8, color=MUTED)
fig.tight_layout(rect=(0.07, 0.025, 0.99, 0.965), h_pad=0.85)
save_figure(fig, "FIG01_Global_SHAP_Beeswarm_5Classes.png")

# ---------------- FIGURE 2: GLOBAL SHAP BAR ------------------------
# Macro-average across five class outputs: each class contributes equally.
# This is a descriptive multiclass summary, not a probability percentage.
class_importance = np.mean(np.abs(shap_values), axis=0)  # features x classes
global_importance = class_importance.mean(axis=1)
bar_df = pd.DataFrame({
    "Feature": predictors,
    "Predictor": [feature_label(f) for f in predictors],
    "MeanAbsSHAP_macro_average": global_importance,
})
bar_df = bar_df.sort_values("MeanAbsSHAP_macro_average", ascending=False)
bar_df.to_csv(TAB / "Global_SHAP_Mean_Absolute_Importance.csv", index=False)
top_bar = bar_df.head(N_GLOBAL).sort_values("MeanAbsSHAP_macro_average")

fig, ax = plt.subplots(figsize=(9.4, 6.2), constrained_layout=True)
bar_colors = [
    "#4776A8", "#287C78", "#D19A36", "#C66D4F", "#80618F",
    "#68838D", "#6D936F", "#B88758", "#547D9A", "#A66A75"
][:len(top_bar)]
bars = ax.barh(
    top_bar["Predictor"], top_bar["MeanAbsSHAP_macro_average"],
    color=bar_colors, height=0.68, edgecolor=PAPER, linewidth=1.0
)
for bar, val in zip(bars, top_bar["MeanAbsSHAP_macro_average"]):
    ax.text(val, bar.get_y() + bar.get_height()/2,
            f"  {val:.3f}", va="center", ha="left",
            fontsize=8, color=MUTED)
ax.set_xlabel("Macro-averaged mean |SHAP| (LightGBM class-output units)")
ax.set_ylabel("Environmental predictor")
ax.set_title("Global importance", loc="left", fontsize=14, pad=12)
ax.text(0, 1.025,
        "Equal-weight average across the five LightGBM class outputs",
        transform=ax.transAxes, fontsize=8, color=MUTED)
style_axis(ax)
fig.text(0.01, -0.025,
         "Values are mean absolute attributions averaged equally across classes; they are not percentages.",
         fontsize=7.7, color=MUTED)
save_figure(fig, "FIG02_Global_SHAP_Bar.png")

# ---------------- FIGURE 3: REPRESENTATIVE WATERFALL ---------------
# Select a correctly classified Very High microcatchment from the existing
# data: closest to the class-5 multivariate median among correctly predicted
# class-5 examples. This avoids hand-picking an extreme case.
p_class5 = prob[:, model_classes.index(5)]
pred5 = predicted == 5
true5 = y == 5
candidates = np.where(pred5 & true5)[0]
selection_rule = "correctly classified class-5 microcatchment closest to the class-5 feature median"
if len(candidates) == 0:
    candidates = np.where(pred5)[0]
    selection_rule = "predicted class-5 microcatchment closest to the class-5 feature median; not necessarily correct"
if len(candidates) == 0:
    raise RuntimeError("LightGBM predicted no Very High microcatchments; cannot select representative case.")

class5_rows = np.where(y == 5)[0]
median_vector = X.iloc[class5_rows].median(axis=0).to_numpy()
candidate_matrix = X.iloc[candidates].to_numpy()
scale = np.nanstd(X.to_numpy(), axis=0)
scale[~np.isfinite(scale) | (scale == 0)] = 1.0
distances = np.sqrt(np.mean(((candidate_matrix - median_vector) / scale) ** 2, axis=1))
selected_idx = int(candidates[int(np.argmin(distances))])
selected_basin = int(data.iloc[selected_idx]["basin_id"])
selected_node = int(data.iloc[selected_idx]["node_index"])
selected_pred = int(predicted[selected_idx])
selected_actual = int(y[selected_idx])

local_exp = shap.Explanation(
    values=shap_values[selected_idx, :, 4],
    base_values=float(base_values[4]),
    data=X.iloc[selected_idx].to_numpy(),
    feature_names=[feature_label(f) for f in predictors],
)
fig = plt.figure(figsize=(10.5, 7.0))
shap.plots.waterfall(local_exp, max_display=12, show=False)

# Refine the waterfall panel only: remove the numeric contribution labels
# printed beside the right ends of the bars, while retaining feature values,
# feature names, f(x), and E[f(X)].
import re
waterfall_ax = plt.gca()
for txt in list(waterfall_ax.texts):
    label = txt.get_text().strip().replace("−", "-")
    if re.fullmatch(r"[+-]\s*\d+(?:\.\d+)?", label):
        txt.remove()

# High-contrast contribution colours: vivid red = positive, electric blue = negative.
POSITIVE_SHAP = "#E60026"
NEGATIVE_SHAP = "#0057FF"
for patch in waterfall_ax.patches:
    try:
        rgba = patch.get_facecolor()
        if len(rgba) >= 3 and rgba[0] >= rgba[2]:
            patch.set_facecolor(POSITIVE_SHAP)
            patch.set_edgecolor(POSITIVE_SHAP)
        else:
            patch.set_facecolor(NEGATIVE_SHAP)
            patch.set_edgecolor(NEGATIVE_SHAP)
    except Exception:
        pass

# Black typography for clear print and journal reproduction.
for txt in waterfall_ax.texts:
    txt.set_color("#000000")
waterfall_ax.tick_params(axis="both", colors="#000000")
waterfall_ax.xaxis.label.set_color("#000000")
waterfall_ax.yaxis.label.set_color("#000000")
plt.title(
    f"Local explanation | Basin {selected_basin}",
    loc="left", fontsize=13, pad=18, color="#000000"
)
plt.gcf().text(
    0.12, 0.94,
    f"Actual class: {CLASS_NAMES[selected_actual]}  •  "
    f"Predicted class: {CLASS_NAMES[selected_pred]}  •  "
    f"P(class 5): {p_class5[selected_idx]:.3f}",
    fontsize=8.5, color="#000000"
)
plt.gcf().text(
    0.12, 0.025,
    "SHAP attribution is in model-output units; P(class 5) is shown separately.",
    fontsize=7.7, color="#000000"
)
save_figure(plt.gcf(), "FIG03_Local_Waterfall_Representative_VeryHigh.png")

local_record = pd.DataFrame([{
    "basin_id": selected_basin,
    "node_index": selected_node,
    "actual_class": selected_actual,
    "predicted_class": selected_pred,
    "predicted_probability_class5": p_class5[selected_idx],
    "selection_rule": selection_rule,
}])
local_record.to_csv(TAB / "Representative_Local_Explanation_Record.csv", index=False)

# ---------------- FIGURE 4: CLASS-5 DEPENDENCE ----------------------
# Choose top four features by class-5 mean absolute SHAP. Predictor values
# are scaled values in the Step-1 modelling matrix; label axis accordingly.
class5_importance = np.mean(np.abs(shap_values[:, :, 4]), axis=0)
dep_idx = np.argsort(class5_importance)[-N_DEPENDENCE:][::-1]
dep_features = [predictors[i] for i in dep_idx]

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
axes = axes.ravel()
for ax, feat_idx in zip(axes, dep_idx):
    xval = X.iloc[:, feat_idx].to_numpy()
    sval = shap_values[:, feat_idx, 4]
    # Colour by the second most important class-5 feature where possible;
    # otherwise colour by this feature's own value.
    interaction_idx = next((j for j in dep_idx if j != feat_idx), feat_idx)
    color_value = X.iloc[:, interaction_idx].to_numpy()
    lo, hi = np.nanpercentile(color_value, [2, 98])
    if hi <= lo:
        c_norm = np.full(len(color_value), 0.5)
    else:
        c_norm = np.clip((color_value - lo) / (hi - lo), 0, 1)
    scatter = ax.scatter(
        xval, sval, c=c_norm, cmap=FEATURE_CMAP, vmin=0, vmax=1,
        s=34, alpha=0.88, edgecolors="#26343A", linewidths=0.18,
        rasterized=True
    )
    # The panels share one common relative Low–High colour scale.
    ax.set_xlabel(f"{feature_label(predictors[feat_idx])} (scaled value)")
    ax.set_ylabel("SHAP value (class-5 output)")
    ax.set_title(feature_label(predictors[feat_idx]),
                 loc="left", fontsize=10, color=CLASS_COLORS[5])
    style_axis(ax, "y")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
fig.suptitle("Class 5 dependence",
             x=0.07, ha="left", y=0.995, fontsize=14, fontweight="bold")
fig.text(0.07, 0.967,
         "Top four predictors • one point per microcatchment",
         fontsize=8.2, color=MUTED)
fig.text(0.07, 0.012,
         "Scaled predictors; associations are not causal thresholds.",
         fontsize=7.7, color=MUTED)
# Reserve a clean right margin for one slim shared colourbar.
fig.tight_layout(rect=(0.06, 0.04, 0.90, 0.94), h_pad=1.35, w_pad=2.0)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
shared_mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=FEATURE_CMAP)
shared_mappable.set_array([])
cax = fig.add_axes([0.925, 0.245, 0.012, 0.52])
cbar = fig.colorbar(shared_mappable, cax=cax)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Low", "Mid", "High"])
cbar.ax.tick_params(labelsize=7, length=0, pad=3)
cbar.set_label("Relative value of interacting predictor", fontsize=7.5,
               labelpad=7, color=MUTED)
cbar.outline.set_visible(False)
save_figure(fig, "FIG04_SHAP_Dependence_VeryHigh_Class5.png")

# ------------------------- AUDIT + README --------------------------
audit = pd.DataFrame({
    "raw_predictor": predictors,
    "publication_label": [feature_label(f) for f in predictors],
})
audit.to_csv(TAB / "Predictor_Label_Audit.csv", index=False)

readme = f"""STEP 05 — FOUR STANDARD SHAP FIGURES
Model: LightGBM
Target: VGAE_FSM5 (codes 1–5)
Microcatchments explained: {len(data)}
Predictors: {len(predictors)}

FIG01: Multiclass beeswarm; five panels, one per class, top 10 predictors.
FIG02: Global bar; macro-average mean absolute SHAP across five classes.
FIG03: Local class-5 waterfall for basin_id {selected_basin}, node_index {selected_node}.
Selection: {selection_rule}.
Actual class: {selected_actual}; predicted class: {selected_pred}.
Predicted probability for class 5: {p_class5[selected_idx]:.6f}.
FIG04: Four class-5 dependence plots.

SCIENTIFIC LIMITATIONS
1. SHAP explains the LightGBM surrogate approximation of VGAE-derived
   susceptibility labels, not the internal VGAE model and not observed flood events.
2. SHAP attributions are not causal effects.
3. Waterfall uses LightGBM class-output units. The class-5 probability
   is displayed separately and is not the waterfall endpoint.
4. The input modelling matrix contains scaled predictors. Dependence plots
   therefore show scaled predictor values, not physical units or thresholds.
5. This script explains the existing 508-row dataset and does not refit the
   surrogate. Interpretive fidelity should be reported alongside Step 3
   held-out performance; random holdout does not prove spatial transferability.
6. Macro-average mean |SHAP| is a descriptive equal-weight summary across
   class outputs, not a percentage of total influence.
7. The representative local case is selected by a reproducible rule, not
   chosen because it has an extreme score.
"""
(OUT / "README_STEP05_SHAP_LIMITATIONS.txt").write_text(readme, encoding="utf-8")

print("\nSTEP 05 COMPLETE")
print("Figures:", FIG)
print("Tables:", TAB)
print(f"Representative microcatchment: {selected_basin} (node {selected_node})")
print(f"Actual/predicted class: {selected_actual}/{selected_pred}")
print(f"P(class 5): {p_class5[selected_idx]:.4f}")
