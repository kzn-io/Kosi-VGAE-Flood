# -*- coding: utf-8 -*-
"""
KRB SPATIAL SHAP ATLAS — CLASS 5 (VERY HIGH)
Maps signed SHAP contribution and absolute contribution for the six
most important predictors in the LightGBM surrogate's Class-5 output.

Scientific scope:
  SHAP values explain the LightGBM surrogate approximation of VGAE_FSM5
  labels. They are not causal effects and are not observed flood measurements.

Input files (place beside this script OR edit the paths below):
  LightGBM_SHAP_LONG_ALL_508.csv
  KRB_MicroCatchments_ALL_GAPS_FILLED.shp (+ .shx/.dbf/.prj)
  The supplied polygon layer is joined using MICRO_ID (1–508) to SHAP basin_id (1–508).

Outputs:
  KRB_SPATIAL_SHAP_CLASS5/
    FIGURES/ 12 single-factor maps + two six-panel atlas maps (PNG/PDF)
    TABLES/   mapped values CSV, predictor ranking CSV, QC report
    VECTOR/   joined GeoPackage with all mapped SHAP fields
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ PATHS ---------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = PROJECT_ROOT / "data"
SHAP_CSV_NAME = "LightGBM_SHAP_LONG_ALL_508.csv"
SHAP_CSV_OVERRIDE = DATA_DIR / "processed" / SHAP_CSV_NAME
POLYGON_OVERRIDE = DATA_DIR / "spatial_units" / "krb_microcatchment_boundaries_508.shp"

# Keep output paths short: deeply nested Windows paths can exceed legacy path limits.
OUT = PROJECT_ROOT / "results" / "spatial_shap_class5"
FIG = OUT / "FIGURES"
TAB = OUT / "TABLES"
VEC = OUT / "VECTOR"
for folder in (OUT, FIG, TAB, VEC):
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------------ SETTINGS ------------------------------------
TARGET_CLASS = 5
N_FACTORS = 6
EXPECTED_N = 508
MAP_DPI = 600
FIGSIZE_SINGLE = (7.2, 6.2)
FIGSIZE_ATLAS = (18.0, 12.0)  # journal landscape: 3 columns x 2 rows
MAP_BG = "#FFFFFF"
INK = "#111111"
BOUNDARY = "#252525"
# Strong, colour-blind-conscious divergent ramp: blue = negative, red = positive.
SIGNED_CMAP = "RdBu_r"
# Magnitude ramp: pale yellow through orange to dark purple.
MAGNITUDE_CMAP = "magma"
# Percentile clipping controls outlier domination; clipped values remain in data.
CLIP_PERCENTILE = 98

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": MAP_BG,
    "axes.facecolor": MAP_BG,
    "savefig.facecolor": MAP_BG,
})

def resolve_input(override, filename, candidates=None):
    if override:
        p = Path(override).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Configured input does not exist: {p}")
        return p
    p = HERE / filename
    if p.exists():
        return p
    for root in (HERE, PROJECT_ROOT, DATA_DIR):
        direct = root / filename
        if direct.exists():
            return direct
    if candidates:
        for name in candidates:
            q = HERE / name
            if q.exists():
                return q
    raise FileNotFoundError(f"Missing input: {filename}. Set its path at the top of this script.")

def safe_stem(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)

def save_map(fig, stem):
    png_path = FIG / f"{stem}.png"
    pdf_path = FIG / f"{stem}.pdf"
    try:
        fig.savefig(png_path, dpi=MAP_DPI, bbox_inches="tight",
                    facecolor=MAP_BG, pil_kwargs={"compress_level": 6})
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=MAP_BG)
    except OSError as exc:
        plt.close(fig)
        raise OSError(
            f"Could not save map output. Attempted path: {png_path if not png_path.exists() else pdf_path}. "
            "The script now uses shortened output paths and filenames. "
            "Check that the output folder is writable and that the file is not open in another program."
        ) from exc
    plt.close(fig)

def style_map_ax(ax):
    ax.set_axis_off()
    ax.set_facecolor(MAP_BG)

def plot_single(gdf, field, title, mode, outstem):
    values = pd.to_numeric(gdf[field], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite map values in {field}.")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    style_map_ax(ax)
    if mode == "signed":
        lim = float(np.nanpercentile(np.abs(values), CLIP_PERCENTILE))
        lim = max(lim, 1e-12)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        gdf.plot(column=field, ax=ax, cmap=SIGNED_CMAP, norm=norm,
                 linewidth=0.12, edgecolor="#777777",
                 missing_kwds={"color": "#EEEEEE"})
        sm = ScalarMappable(norm=norm, cmap=SIGNED_CMAP)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          fraction=0.045, pad=0.025, shrink=0.78, aspect=38)
        cb.set_label("Signed SHAP", fontsize=9)
        cb.set_ticks([-lim, 0, lim])
        cb.set_ticklabels([f"≤ { -lim:.2f}", "0", f"≥ {lim:.2f}"])
    else:
        vmax = float(np.nanpercentile(values, CLIP_PERCENTILE))
        vmax = max(vmax, 1e-12)
        norm = Normalize(vmin=0, vmax=vmax)
        gdf.plot(column=field, ax=ax, cmap=MAGNITUDE_CMAP, norm=norm,
                 linewidth=0.12, edgecolor="#777777",
                 missing_kwds={"color": "#EEEEEE"})
        sm = ScalarMappable(norm=norm, cmap=MAGNITUDE_CMAP)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          fraction=0.045, pad=0.025, shrink=0.78, aspect=38)
        cb.set_label("|SHAP|", fontsize=9)
        cb.set_ticks([0, vmax/2, vmax])
    ax.set_title(title, color=INK, pad=12)
    ax.text(0.01, 0.015, "Kosi River Basin | 508 microcatchments | Class 5: Very High",
            transform=ax.transAxes, fontsize=7.5, color=INK,
            ha="left", va="bottom")
    save_map(fig, outstem)

def plot_atlas(gdf, selected, mode, stem, title):
    # Wide, publication-style atlas: three columns by two rows.
    ncols = 3
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=FIGSIZE_ATLAS,
        gridspec_kw={"wspace": 0.12, "hspace": 0.22}
    )
    axes = np.asarray(axes).ravel()

    for panel_index, (ax, (feature, label, field)) in enumerate(zip(axes, selected), start=1):
        style_map_ax(ax)
        vals = pd.to_numeric(gdf[field], errors="coerce").to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Non-finite values found in atlas field: {field}")

        if mode == "signed":
            lim = max(float(np.nanpercentile(np.abs(vals), CLIP_PERCENTILE)), 1e-12)
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
            cmap = SIGNED_CMAP
            legend_label = "Signed SHAP"
        else:
            lim = max(float(np.nanpercentile(vals, CLIP_PERCENTILE)), 1e-12)
            norm = Normalize(vmin=0, vmax=lim)
            cmap = MAGNITUDE_CMAP
            legend_label = "|SHAP|"

        gdf.plot(
            column=field, ax=ax, cmap=cmap, norm=norm,
            linewidth=0.13, edgecolor="#777777",
            missing_kwds={"color": "#EEEEEE", "edgecolor": "#777777"}
        )

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(
            sm, ax=ax, orientation="horizontal",
            fraction=0.050, pad=0.035, shrink=0.88, aspect=38
        )
        cb.ax.tick_params(labelsize=7.5, colors=INK, length=2)
        cb.set_label(legend_label, fontsize=9.5, color=INK, labelpad=3)
        ax.set_title(f"{label}", fontsize=11.5, fontweight="bold",
                     color=INK, pad=8)

        # Small panel index for clear reading order.
        ax.text(
            0.015, 0.985, f"({chr(96 + panel_index)})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold", color=INK
        )

    for ax in axes[len(selected):]:
        ax.remove()

    fig.suptitle(title, fontsize=18, fontweight="bold", color=INK, y=0.975)
    fig.text(
        0.5, 0.018,
        "Surrogate attribution; not causal or observed flood probability.",
        ha="center", va="bottom", fontsize=8.5, color=INK
    )

    # Explicit margins preserve large map panels and prevent clipped titles/colourbars.
    fig.subplots_adjust(
        left=0.025, right=0.985, top=0.925, bottom=0.075,
        wspace=0.12, hspace=0.24
    )
    save_map(fig, stem)

# ------------------------------ LOAD & QC -----------------------------------
shap_path = resolve_input(SHAP_CSV_OVERRIDE, SHAP_CSV_NAME)
if POLYGON_OVERRIDE:
    polygon_path = Path(POLYGON_OVERRIDE).expanduser()
    if not polygon_path.exists():
        raise FileNotFoundError(f"Polygon file not found: {polygon_path}")
else:
    polygon_path = None
    for candidate in (DATA_DIR / "spatial_units" / "krb_microcatchment_boundaries_508.shp",
                      DATA_DIR / "spatial_units" / "krb_microcatchment_feature_attributes_508.gpkg"):
        if candidate.exists():
            polygon_path = candidate
            break
    if polygon_path is None:
        raise FileNotFoundError("Microcatchment layer not found in data/spatial_units.")

print(f"SHAP table: {shap_path}")
print(f"Polygons:   {polygon_path}")
long = pd.read_csv(shap_path)

required = {"basin_id", "node_index", "explained_class_code",
            "feature", "feature_value", "shap_value"}
missing = required.difference(long.columns)
if missing:
    raise ValueError(f"SHAP CSV is missing required columns: {sorted(missing)}")

long["basin_id"] = pd.to_numeric(long["basin_id"], errors="raise").astype(int)
long["explained_class_code"] = pd.to_numeric(
    long["explained_class_code"], errors="raise").astype(int)
long["shap_value"] = pd.to_numeric(long["shap_value"], errors="coerce")
long["feature_value"] = pd.to_numeric(long["feature_value"], errors="coerce")
class5 = long[long["explained_class_code"] == TARGET_CLASS].copy()
if class5.empty:
    raise ValueError("No Class-5 SHAP records found.")

# Select six factors by Class-5 mean absolute SHAP across all 508 units.
ranking = (class5.groupby("feature", as_index=False)
           .agg(mean_abs_shap=("shap_value", lambda s: s.abs().mean()),
                mean_signed_shap=("shap_value", "mean"),
                valid_units=("basin_id", "nunique")))
ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
selected_features = ranking.head(N_FACTORS)["feature"].tolist()
if len(selected_features) != N_FACTORS:
    raise ValueError(f"Expected {N_FACTORS} predictors; found {len(selected_features)}.")

# Verify one record per basin-feature for the selected class.
subset = class5[class5["feature"].isin(selected_features)].copy()
dups = subset.duplicated(["basin_id", "feature"], keep=False)
if dups.any():
    examples = subset.loc[dups, ["basin_id", "feature"]].head(10).to_dict("records")
    raise ValueError(
        "Duplicate Class-5 SHAP rows for basin_id-feature combinations. "
        f"Examples: {examples}"
    )
if subset["shap_value"].isna().any():
    raise ValueError("Selected Class-5 SHAP records contain missing/non-numeric shap_value.")

# Each selected factor must have one SHAP value for every one of the 508 nodes.
coverage = subset.groupby("feature")["basin_id"].nunique()
bad_coverage = coverage[coverage != EXPECTED_N]
if not bad_coverage.empty:
    raise ValueError(
        "Selected predictors do not each cover all 508 SHAP basin IDs: "
        f"{bad_coverage.to_dict()}"
    )

# Wide format: one signed SHAP field per selected predictor.
wide = subset.pivot(index="basin_id", columns="feature", values="shap_value")
if wide.index.duplicated().any():
    raise ValueError("Duplicate basin IDs in pivoted SHAP table.")
if len(wide) != EXPECTED_N or set(wide.index) != set(range(1, EXPECTED_N + 1)):
    raise ValueError("Selected Class-5 SHAP data do not contain exactly basin IDs 1–508.")

# Load polygon layer and identify join key.
polygons = gpd.read_file(polygon_path)
if len(polygons) != EXPECTED_N:
    raise ValueError(f"Expected {EXPECTED_N} polygons; found {len(polygons)}.")
if polygons.crs is None:
    raise ValueError("Polygon layer has no CRS. Define its true CRS before mapping.")
# IMPORTANT: MICRO_ID is the verified 1–508 microcatchment identifier.
# BASIN_ID is a different numbering field in this shapefile (1–509 with a gap),
# so it must NOT be used for this SHAP-table join.
join_key = "MICRO_ID"
if join_key not in polygons.columns:
    raise ValueError(
        f"Required polygon join field '{join_key}' is missing. "
        f"Available fields: {list(polygons.columns)}"
    )

polygons["basin_id"] = pd.to_numeric(
    polygons[join_key], errors="coerce"
)
if polygons["basin_id"].isna().any():
    bad_n = int(polygons["basin_id"].isna().sum())
    raise ValueError(
        f"{bad_n} polygon MICRO_ID values are missing or non-numeric."
    )
if not np.allclose(polygons["basin_id"], np.round(polygons["basin_id"])):
    raise ValueError("Polygon MICRO_ID contains non-integer values.")
polygons["basin_id"] = polygons["basin_id"].astype(int)

if polygons["basin_id"].duplicated().any():
    duplicate_ids = sorted(
        polygons.loc[polygons["basin_id"].duplicated(False), "basin_id"].unique().tolist()
    )
    raise ValueError(f"Duplicate polygon MICRO_ID values: {duplicate_ids[:20]}")

expected_ids = set(range(1, EXPECTED_N + 1))
polygon_ids = set(polygons["basin_id"].tolist())
if polygon_ids != expected_ids:
    missing_ids = sorted(expected_ids - polygon_ids)
    unexpected_ids = sorted(polygon_ids - expected_ids)
    raise ValueError(
        "Polygon MICRO_ID values do not match the required 1–508 sequence. "
        f"Missing IDs (first 20): {missing_ids[:20]}; "
        f"unexpected IDs (first 20): {unexpected_ids[:20]}."
    )

# Confirm the SHAP table and polygon layer use the same identifier domain.
shap_ids = set(wide.index.astype(int).tolist())
if shap_ids != polygon_ids:
    missing_shapes = sorted(shap_ids - polygon_ids)
    missing_shap = sorted(polygon_ids - shap_ids)
    raise ValueError(
        "SHAP-to-polygon ID mismatch. "
        f"SHAP IDs without polygons (first 20): {missing_shapes[:20]}; "
        f"polygon IDs without SHAP values (first 20): {missing_shap[:20]}."
    )

# Ensure a projected CRS for scale bar/area interpretation; preserve source CRS.
if not polygons.crs.is_projected:
    raise ValueError(
        f"Expected projected CRS for the KRB polygons; found {polygons.crs}. "
        "Reproject to the verified study CRS before running."
    )
if polygons.geometry.isna().any() or polygons.geometry.is_empty.any():
    raise ValueError("Null or empty polygon geometries found.")
invalid_count = int((~polygons.geometry.is_valid).sum())
if invalid_count:
    print(f"WARNING: {invalid_count} invalid geometries; repairing with make_valid/buffer(0).")
    try:
        polygons.geometry = polygons.geometry.make_valid()
    except Exception:
        polygons.geometry = polygons.geometry.buffer(0)
if polygons.geometry.isna().any() or polygons.geometry.is_empty.any():
    raise ValueError("Geometry repair produced empty/null features.")

merged = polygons.merge(wide, on="basin_id", how="left", validate="one_to_one")
if merged[list(selected_features)].isna().any().any():
    raise ValueError("Some polygons did not receive all selected SHAP values.")
if len(merged) != EXPECTED_N:
    raise ValueError("Join changed the expected polygon count.")

# Predictor names for readable map titles.
def label_feature(f):
    endings = [("_scaled_mean", ""), ("_scaled_std", " variability"), ("_fraction", " share")]
    base, suffix = f, ""
    for ending, sfx in endings:
        if base.endswith(ending):
            base, suffix = base[:-len(ending)], sfx
            break
    labels = {
        "DEM": "Elevation", "Mean_Rainfall": "Mean rainfall",
        "ET": "Evapotranspiration", "Max1Day_Rainfall": "Maximum 1-day rainfall",
        "SMAP": "Surface soil moisture (SMAP)", "Sand_0_5cm": "Sand content (0–5 cm)",
        "Clay_0_5cm": "Clay content (0–5 cm)", "Silt_0_5cm": "Silt content (0–5 cm)",
        "SPI": "Standardised precipitation index", "Slope": "Terrain slope",
        "TWI": "Topographic wetness index",
        "Distance_to_Main_River": "Distance to main river",
        "Drainage_Density": "Drainage density",
    }
    return labels.get(base, base.replace("_", " ")) + suffix

selected = []
rank_rows = []
for rank, feature in enumerate(selected_features, start=1):
    label = label_feature(feature)
    signed_field = f"S{rank:02d}"
    abs_field = f"A{rank:02d}"
    merged[signed_field] = merged[feature].astype(float)
    merged[abs_field] = merged[signed_field].abs()
    selected.append((feature, label, signed_field))
    rr = ranking.loc[ranking["feature"] == feature].iloc[0]
    rank_rows.append({
        "rank": rank, "feature": feature, "publication_label": label,
        "class5_mean_absolute_shap": float(rr["mean_abs_shap"]),
        "class5_mean_signed_shap": float(rr["mean_signed_shap"]),
        "microcatchments": int(rr["valid_units"]),
        "signed_map_field": signed_field, "absolute_map_field": abs_field,
    })

# Export ranking and spatial joined values.
pd.DataFrame(rank_rows).to_csv(TAB / "Class5_Spatial_SHAP_Predictor_Ranking.csv", index=False)
export_fields = ["basin_id"] + [x[2] for x in selected] + [
    f"A{i:02d}" for i in range(1, len(selected) + 1)
]
merged.drop(columns=["geometry"]).to_csv(
    TAB / "Class5_Spatial_SHAP_Values_508_Microcatchments.csv", index=False
)
gpkg_path = VEC / "KRB_Class5_Spatial_SHAP_508_Microcatchments.gpkg"
if gpkg_path.exists():
    gpkg_path.unlink()
merged[["basin_id", "geometry"] + [f"S{i:02d}" for i in range(1, 7)] +
       [f"A{i:02d}" for i in range(1, 7)]].to_file(
    gpkg_path, layer="class5_spatial_shap", driver="GPKG"
)

# ------------------------------ MAPS ----------------------------------------
for rank, (feature, label, signed_field) in enumerate(selected, start=1):
    abs_field = f"A{rank:02d}"
    # Short filename avoids Windows path/filename limitations; full predictor
    # name remains in the map title, atlas panel title and ranking CSV.
    short_name = safe_stem(feature)[:24]
    stem = f"C5_{rank:02d}_{short_name}"
    plot_single(
        merged, signed_field,
        f"Class 5 spatial SHAP contribution — {label}",
        "signed", stem + "_SIGNED"
    )
    plot_single(
        merged, abs_field,
        f"Class 5 spatial SHAP magnitude — {label}",
        "magnitude", stem + "_MAGNITUDE"
    )

# Two publication atlas sheets: signed and absolute contributions.
plot_atlas(
    merged, selected, "signed", "CLASS5_ATLAS_SIGNED",
    "Spatial SHAP contributions | Very High susceptibility (Class 5)"
)
abs_selected = [(f, lab, f"A{i:02d}") for i, (f, lab, _) in enumerate(selected, start=1)]
plot_atlas(
    merged, abs_selected, "magnitude", "CLASS5_ATLAS_MAGNITUDE",
    "Spatial SHAP contribution magnitude | Very High susceptibility (Class 5)"
)

# QC report
qc = [
    "KRB SPATIAL SHAP CLASS-5 — QUALITY CONTROL",
    f"SHAP source: {shap_path}",
    f"Polygon source: {polygon_path}",
    f"Target class: {TARGET_CLASS} (Very High)",
    f"Expected microcatchments: {EXPECTED_N}",
    f"Joined microcatchments: {len(merged)}",
    f"Polygon CRS: {merged.crs}",
    f"Polygon join key: {join_key} (MICRO_ID 1–508 joined to SHAP basin_id 1–508)",
    f"Selected predictors: {', '.join(selected_features)}",
    f"Invalid geometries repaired: {invalid_count}",
    f"Missing mapped SHAP values: {int(merged[[x[2] for x in selected]].isna().sum().sum())}",
    "Join validation: one-to-one; polygon MICRO_ID matched to SHAP basin_id.",
    "Signed maps: zero-centred diverging scale; percentile clipping affects display only.",
    "Magnitude maps: absolute SHAP values; percentile clipping affects display only.",
    "Atlas panels are independently normalized per predictor; compare spatial patterns, not raw colour intensity across factors.",
    "Interpretation: LightGBM surrogate explanations of VGAE-derived labels; not causal effects or observed flood occurrence.",
]
(TAB / "SPATIAL_SHAP_QC_REPORT.txt").write_text("\n".join(qc), encoding="utf-8")

print("\nSPATIAL SHAP COMPLETE")
print(f"Output directory: {OUT}")
print(f"Mapped microcatchments: {len(merged)}")
print("Selected Class-5 predictors:")
for r in rank_rows:
    print(f"  {r['rank']}. {r['publication_label']} | mean |SHAP|={r['class5_mean_absolute_shap']:.6f}")
print("Figures: 12 single-factor maps + 2 six-factor atlas maps (PNG and PDF).")
print("Vector: joined GeoPackage with signed and absolute SHAP fields.")
