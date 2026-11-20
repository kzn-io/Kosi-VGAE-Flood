# Data dictionary

## Main feature matrix

`data/processed/KRB_508_Basin_Feature_Matrix.csv`

- `basin_id`: basin/microcatchment identifier used by the feature and SHAP tables.
- `*_scaled_mean`: scaled mean summary for a predictor within a microcatchment.
- `*_scaled_std`: scaled standard-deviation summary for a predictor within a microcatchment.
- `WorldCover_*_fraction`: fraction of the microcatchment assigned to the corresponding ESA WorldCover class code.

The matrix contains 508 records and 38 columns (one identifier and 37 feature columns).

## SHAP long table

`data/processed/LightGBM_SHAP_LONG_ALL_508.csv`

| Field | Meaning |
|---|---|
| `basin_id` | Microcatchment identifier; spatially joins to polygon `MICRO_ID`. |
| `node_index` | Zero-based graph node index. |
| `actual_VGAE_FSM5` | VGAE-derived five-class label used as surrogate target. |
| `predicted_class` | Surrogate model prediction. |
| `explained_class_code` | Class output whose SHAP contribution is recorded. |
| `feature` | Predictor name. |
| `feature_value` | Predictor value used by the surrogate. |
| `shap_value` | SHAP contribution in the model output space. |

## Polygon layer

`data/spatial_units/krb_microcatchment_boundaries_508.shp` and its `.shx`, `.dbf`, `.prj`, `.cpg` companions.

- `MICRO_ID`: intended 1–508 join identifier.
- `MACRO_ID`: parent macro-basin identifier.
- `AREA_KM2`: microcatchment area in square kilometres.
- CRS: EPSG:32645.


## Five-class VGAE flood-susceptibility output

`data/flood_susceptibility/vgae_flood_susceptibility_5class_508.gpkg` contains 508 microcatchment polygons and the fields `VGAE_FSM5`, `VGAE_FSM5_name`, `VGAE_FSM_cluster10`, and `physical_susceptibility_index`. The five classes are relative susceptibility categories, not calibrated probabilities.

## Spatial Class-5 SHAP vector

`data/spatial_explainability/class5_spatial_shap_explanations_508.gpkg` contains 508 polygons joined by `MICRO_ID` to `basin_id`. `F01`–`F06` prefixes are defined in `data/spatial_explainability/class5_top_six_shap_feature_ranking.csv`. Each prefix has `_value`, `_shap`, and `_abs_shap` fields for predictor value, signed SHAP contribution, and absolute SHAP magnitude respectively.
