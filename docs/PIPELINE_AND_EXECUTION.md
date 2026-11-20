# Pipeline and execution notes

## Stage A — raster harmonisation

`01_raster_harmonization/01_align_environmental_rasters_to_reference_grid.py` uses `DEM_30m.tif` as the reference grid and lists 12 continuous predictor rasters. It writes aligned float32 rasters, a CSV QC report and reference-grid metadata. Edit `ROOT`, `INPUT_DIR`, `OUTPUT_DIR` and `REFERENCE_RASTER` at the top of the script before use.

The companion hydrological reconstruction sources derive terrain/hydrology products (conditioned DEM, D8 flow direction, accumulation, TWI, stream network, drainage density and distance-to-drainage). Their input/output paths are hard-coded and must be reviewed before execution.

## Stage B — feature matrix and scaling

The supplied `KRB_508_Basin_Feature_Matrix.csv` has 508 rows and 38 columns. It includes scaled mean and standard-deviation fields for continuous predictors and WorldCover fractions. It is the downstream feature matrix used by the graph/FSM work.

**Important provenance boundary:** the recovered source collection did not contain the exact raw zonal-statistics-to-scaled-feature script or the fitted scaler object/parameters. Therefore this repository preserves the resulting scaled matrix but does not claim to reproduce the original scaling transformation from raw raster statistics. To make that stage fully reproducible, add the original raw zonal-statistics table and the exact scaler code/parameters before publication.

## Stage C — QC and graph construction

Run the input audit and alignment scripts before graph construction. The 508-node graph workflows include polygon-touch spatial adjacency. The graph outputs also include a directed hydrological network (495 directed links in the project QC record) and candidate adjacency products. These graph types encode different relationships:

- **Spatial adjacency:** neighbouring microcatchments, generally undirected.
- **Hydrological edges:** directed downstream connectivity.

Keep node order consistent across the feature matrix, polygon layer, mapping table, and edge indices. The spatial SHAP join uses `MICRO_ID` to `basin_id`; the project node convention maps `basin_id = node_index + 1`.

## Stage D — VGAE and five-class target

The available unsupervised FSM stage consumes saved embeddings and graph artefacts. The original VGAE training script was not present in the recovered source archive. Do not claim that the included `10_run_unsupervised_susceptibility_regionalization.py` retrains the VGAE; it reads pre-existing embeddings and performs downstream analysis. Restore the original training source and model configuration to make the embedding stage independently executable.

## Stage E — surrogate and SHAP

Run the scripts in this order:

1. `12_validate_surrogate_model_inputs.py`
2. `13_create_stratified_model_validation_split.py`
3. `14_benchmark_multiclass_surrogate_models.py`
4. `15_calculate_multiclass_shap_explanations.py`
5. `16_export_publication_quality_shap_figures.py`

The SHAP figures are exported at 600 dpi (PNG) and as vector PDF. Step 4 calculates multiclass SHAP for LightGBM, Random Forest and XGBoost; Step 5 produces the selected LightGBM global, local and dependence figures.

These scripts explain the `VGAE_FSM5` pseudo-labels through surrogate classifiers. `basin_id` and `node_index` are identifiers, not predictors. The 80:20 stratified split in the source scripts is not a spatially independent validation design; avoid claims of spatial transferability from that split alone.

## Stage F — spatial SHAP

Run `07_spatial_explainability/17_generate_class5_spatial_shap_atlas.py`. It ranks the six factors by Class-5 mean absolute SHAP and maps signed and absolute SHAP values across the 508 polygons. Its join key is polygon `MICRO_ID` and SHAP `basin_id`. It exports 12 single-factor maps and two six-panel atlases at 600 dpi (PNG) and vector PDF, plus a joined GeoPackage and QC report. Inspect the maps and QC report before publication.
