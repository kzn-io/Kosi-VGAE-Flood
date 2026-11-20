# Reproducibility gaps and provenance

This file intentionally records what is and is not present in the recovered source bundle.

| Component | Present in this bundle | Limitation |
|---|---|---|
| Raster harmonisation | Python source | Requires original raster inputs and path configuration. |
| Hydrological predictor reconstruction | Python source | Requires DEM, slope, masks and basin boundary; paths are hard-coded. |
| Raw zonal statistics to scaled 508-row feature matrix | Derived scaled CSV only | Exact historic scaler code, raw zonal table and fitted parameters were not recovered. |
| Feature / graph QC | Python source | Requires the corresponding inputs and correct path configuration. |
| Spatial adjacency construction | Python source | Verify node order and polygon-feature alignment before model use. |
| Directed hydrological graph | Edge/adjacency outputs | The exact original graph-tracing source was not identified in the recovered code set. |
| VGAE training | Saved project artifacts exist in prior project archive | Original VGAE training source is absent from the recovered scripts; do not represent downstream embedding consumers as training code. |
| Five-class FSM | Python source | Uses saved embeddings and feature inputs. Classes are relative, not calibrated probabilities. |
| Surrogate model and SHAP | Python source and SHAP long table | Explains VGAE-derived pseudo-labels, not observed flood events or causal effects. |
| Spatial SHAP | Python source, polygons and SHAP table | Final execution and visual verification must be completed on the user's machine. |

## Data integrity facts represented in the supplied outputs

- The feature matrix is a 508-row product with 38 columns.
- The SHAP long table contains fields `basin_id`, `node_index`, `actual_VGAE_FSM5`, `predicted_class`, `explained_class_code`, `feature`, `feature_value`, and `shap_value`.
- The supplied polygon layer has 508 microcatchments. The intended join is `MICRO_ID` (1–508) to SHAP `basin_id` (1–508).
- The supplied polygon layer is projected in EPSG:32645.

These are dataset-specific checks, not guarantees that every script will run unchanged in a fresh environment.
