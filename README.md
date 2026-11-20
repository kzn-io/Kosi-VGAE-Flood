# Kosi River Basin Flood-Susceptibility Research Pipeline
[![DOI](https://zenodo.org/badge/1465427899.svg)](https://doi.org/10.5281/zenodo.24397790)

Research-code bundle for the complete recovered 508-microcatchment, graph-based flood-susceptibility workflow, from raster harmonisation and feature engineering through QC, graph construction, VGAE-derived susceptibility classes, clustering, surrogate benchmarking and SHAP interpretation. All 17 included Python scripts now use consistent sequential, descriptive filenames.

> **Reproducibility note.** This repository is a curated research-code bundle assembled from the available project scripts and outputs. It is not a claim that every historic intermediate script or raw input raster has been recovered. Several original scripts contain Windows-specific paths and require path configuration before execution. The original VGAE training source and the exact raw-feature-to-scaled-feature transformation script were not present in the recovered files; these gaps are documented below rather than filled with invented provenance.

## Study design at a glance

- **Spatial units:** 508 microcatchments.
- **Polygon identifier used in the spatial SHAP join:** `MICRO_ID` (1–508).
- **SHAP table identifier:** `basin_id` (1–508); the supplied node convention is `basin_id = node_index + 1`.
- **Polygon CRS:** EPSG:32645 in the supplied boundary layer.
- **Susceptibility target explained by the surrogate:** `VGAE_FSM5`, five relative classes (1 = Very Low through 5 = Very High).
- **Predictors:** 34 environmental predictors in the SHAP modelling table; the basin feature matrix includes scaled mean/std summaries and ESA WorldCover class fractions.
- **Interpretation:** the LightGBM model is a surrogate for VGAE-derived labels. SHAP explains the surrogate's output, not observed flood occurrence and not causal effects.

## Repository layout

| Directory | Contents |
|---|---|
| `01_raster_harmonization/` | Predictor raster alignment to the DEM reference grid and hydrological predictor reconstruction scripts. |
| `02_feature_engineering/` | Hydrological-factor reconstruction and feature preparation source. |
| `03_quality_control/` | Input, geometry, node-order and feature/graph alignment audits. |
| `04_graph_construction/` | 508-node spatial adjacency construction scripts. |
| `05_vgae_and_clustering/` | Available unsupervised FSM stage, clustering evaluation and five-class mapping scripts. |
| `06_surrogate_and_shap/` | SHAP input QC, train/test split, surrogate benchmarking, SHAP calculation and figure generation. |
| `07_spatial_explainability/` | Class-5 signed and absolute spatial SHAP atlas script. |
| `data/processed/` | Derived feature matrix and SHAP tables supplied for this research bundle. |
| `data/flood_susceptibility/` | Five-class VGAE-FSM GeoPackage and multi-layer VGAE/DinkNet comparison GeoPackage. |
| `data/spatial_explainability/` | Class-5 SHAP vector GeoPackage, long-form SHAP values and top-six factor lookup. |
| `data/spatial_units/` | Canonically named 508-microcatchment boundary shapefile family and feature-attribute GeoPackage. |
| `data/graph/` | Directed hydrological and candidate adjacency products plus node order. |
| `data/graph_construction/` | Complete 508-microcatchment feature, spatial-unit, graph-topology and outlet-routing validation data package, with standardised filenames. |
| `results/shap_analysis/` | Selected SHAP tables and example figures. |
| `docs/` | Workflow, data dictionary, execution notes and reproducibility limitations. |

## Workflow and execution order

1. **Raster harmonisation** — run `01_raster_harmonization/01_align_environmental_rasters_to_reference_grid.py` to align the listed continuous predictors to the DEM reference grid. The script uses bilinear resampling for continuous layers and nearest-neighbour for categorical layers.
2. **Hydrological predictor reconstruction** — run the relevant reconstruction script in `01_raster_harmonization/` or `02_feature_engineering/` only after checking its configured DEM, slope, mask and basin-boundary paths. These scripts represent different stages/versions; do not run both blindly over the same output directory.
3. **Feature engineering and scaling** — use the derived 508-row feature matrix in `data/processed/KRB_508_Basin_Feature_Matrix.csv` as the supplied scaled feature product. It contains 38 columns: `basin_id` plus 37 predictor summaries/fractions. The exact original script that transformed raw zonal statistics into the `_scaled_mean` / `_scaled_std` columns was not recovered. Do not describe a newly chosen scaler as the historic transformation without validating it against the original raw statistics and saved outputs.
4. **Input QC and node alignment** — run the scripts in `03_quality_control/`. Confirm 508 rows, unique IDs, finite features, and exact row/node correspondence before constructing or training a graph model.
5. **Graph construction** — run the appropriate script in `04_graph_construction/`. The repository preserves spatial adjacency scripts; the supplied `data/graph/` folder also contains directed hydrological edge products. These are different graph concepts and must not be conflated. Review the graph audit and node order before model use.
6. **VGAE / unsupervised susceptibility** — `05_vgae_and_clustering/10_run_unsupervised_susceptibility_regionalization.py` consumes an existing feature matrix, node mapping, graph and saved embeddings. It is not the original VGAE training implementation. The original VGAE training source was not recovered; existing embeddings/model artifacts are not included in this curated code-only bundle.
7. **Surrogate and SHAP** — run the numbered scripts in `06_surrogate_and_shap/` in order: `12_validate_surrogate_model_inputs.py`, `13_create_stratified_model_validation_split.py`, `14_benchmark_multiclass_surrogate_models.py`, `15_calculate_multiclass_shap_explanations.py`, and `16_export_publication_quality_shap_figures.py`. They model the five-class VGAE-derived target and explain the surrogate. The SHAP output table supplied here is `data/processed/LightGBM_SHAP_LONG_ALL_508.csv`. Figures are exported as 600-dpi PNG and vector PDF.
8. **Spatial SHAP atlas** — run `07_spatial_explainability/17_generate_class5_spatial_shap_atlas.py` after setting the input paths to the repository's `data/processed/` and `data/spatial_units/` folders, or copying the CSV and shapefile sidecars beside the script. The join is `MICRO_ID` ↔ `basin_id`.

The canonical boundary dataset is `data/spatial_units/krb_microcatchment_boundaries_508.shp` (with its required sidecar files). The five-class susceptibility output is `data/flood_susceptibility/vgae_flood_susceptibility_5class_508.gpkg`. The spatial Class-5 SHAP vector is `data/spatial_explainability/class5_spatial_shap_explanations_508.gpkg`; its six feature prefixes are documented in `class5_top_six_shap_feature_ranking.csv`. Graph-construction inputs and outlet-routing checks remain documented in `data/graph_construction/README.md`.

Complete renamed script inventory: [`docs/SOURCE_CODE_INDEX.md`](docs/SOURCE_CODE_INDEX.md). Code validation and figure-export standards: [`docs/CODE_QUALITY_AND_PUBLICATION_STANDARDS.md`](docs/CODE_QUALITY_AND_PUBLICATION_STANDARDS.md).

Detailed sequence and known limitations: see [`docs/PIPELINE_AND_EXECUTION.md`](docs/PIPELINE_AND_EXECUTION.md) and [`docs/REPRODUCIBILITY_GAPS.md`](docs/REPRODUCIBILITY_GAPS.md).

## Environment

Recommended: Python 3.11 on Windows or Linux. Install geospatial libraries from a compatible conda-forge environment where possible; PyTorch/PyG wheels must match the local CPU/CUDA and PyTorch versions.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the matching PyTorch and PyTorch Geometric builds separately if you run graph-learning stages. Some scripts additionally require LightGBM, XGBoost, SHAP and joblib.

## Data and GitHub guidance

- The included data are derived project outputs, not the complete raw satellite/raster archive.
- Do not commit licensed third-party rasters, API credentials, personal data, or very large intermediate rasters without checking their redistribution terms.
- If the repository is made public, confirm that the basin boundary and all derived products can be redistributed.
- Add a project-specific licence only after deciding the intended reuse terms. No licence is asserted by this bundle.

## Scientific interpretation

The five classes are relative susceptibility strata, not calibrated flood probabilities. The SHAP stage explains a LightGBM surrogate trained to reproduce `VGAE_FSM5`; it does not convert the unsupervised labels into observed flood labels. SAR/FIM validation is a separate evaluation stream and should be reported independently.


## Principal GIS deliverables

- **Microcatchment boundaries:** `data/spatial_units/krb_microcatchment_boundaries_508.shp` (508 polygons; EPSG:32645).
- **VGAE five-class FSM:** `data/flood_susceptibility/vgae_flood_susceptibility_5class_508.gpkg` (508 features; includes `VGAE_FSM5` and class-name fields).
- **VGAE/DinkNet comparison:** `data/flood_susceptibility/vgae_dinknet_fsm_comparison_layers_508.gpkg` (four thematic layers).
- **Class-5 spatial SHAP vector:** `data/spatial_explainability/class5_spatial_shap_explanations_508.gpkg` (508 polygons joined to signed and absolute SHAP values for the six highest-ranked Class-5 predictors).

The SHAP vector is derived from `explained_class_code == 5` in the supplied long-format LightGBM SHAP table. `F01`–`F06` prefixes are linked to full predictor names and mean absolute SHAP rankings in `class5_top_six_shap_feature_ranking.csv`. The vector contains both predictor values and SHAP contributions; these are distinct fields and must not be interpreted interchangeably.

## SHAP publication outputs

- `results/shap_analysis/step04_multimodel_shap/`: three-model multiclass SHAP tables, figures and run audit.
- `results/shap_analysis/step05_lightgbm_shap/`: selected LightGBM global, local and dependence figures and supporting tables.
- `results/spatial_shap_class5/`: Class-5 signed and absolute SHAP maps, ranking/QC tables and joined GeoPackage.
- Figure exports use 600-dpi PNG for raster workflows and vector PDF for journal layout.

## SAR/FIM multi-event validation (2020–2024)

The `08_sar_fim_validation/` module contains the five-event VGAE-only
leave-one-basin-out (LOBO) peer-cluster analysis, its figure-generation
script, supplied result tables/workbook, and ten publication figures.
The analysis evaluates the two specified observed flood-fraction targets
(>=1% and >=5%) while retaining the original K=10 cluster assignments.
See `08_sar_fim_validation/docs/METHOD_AND_REPRODUCTION.md` for the method,
interpretation limits, and local path configuration.
