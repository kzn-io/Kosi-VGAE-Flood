# Code quality and publication standards

## Repository-wide code naming

All 17 Python scripts in the recovered main workflow have been renamed using a sequential stage identifier and a descriptive action-oriented filename. The sequence covers raster alignment, hydrological reconstruction, feature engineering, QC, graph construction, VGAE-derived FSM, clustering, surrogate benchmarking, SHAP calculation, figure export and spatial SHAP mapping.

## Validation performed for this release

- Python syntax compilation was run against every included `.py` file.
- Result: 17 scripts checked; 0 syntax compilation errors.
- Documentation references were updated to the renamed script paths.
- The original model logic, input paths, feature definitions and model parameters were not silently rewritten as part of the filename standardisation.

Syntax compilation does not establish that a script will run end-to-end. Each script still requires its documented input files, compatible package versions, configured local paths and (for graph-learning stages) matching PyTorch/PyTorch Geometric builds. Run stages in the documented order and inspect the generated QC reports.

## Figure export standard

The SHAP figure and spatial atlas scripts specify 600-dpi PNG export and vector PDF export. Raster figures should be inspected at final journal column width; vector PDFs should be checked for embedded fonts, legible labels and unclipped legends. 600 dpi is an export setting, not a guarantee of scientific or editorial acceptance.

## Scientific safeguards

- Keep spatial adjacency and directed hydrological connectivity as separate graph relations.
- Preserve node-order consistency across features, polygons, graph edges and embeddings.
- The VGAE five-class output is a relative susceptibility classification, not a calibrated probability.
- SHAP describes the LightGBM surrogate's approximation of the VGAE-derived labels; it is not causal evidence or direct observation of flood occurrence.
- The recovered 80:20 stratified holdout is not spatially independent validation.
- The original raw-zonal-statistics-to-scaled-feature transformation and original VGAE training implementation were not recovered. See `REPRODUCIBILITY_GAPS.md`.
