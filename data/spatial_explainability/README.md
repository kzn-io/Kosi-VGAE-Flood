# Spatial explainability data

`class5_spatial_shap_explanations_508.gpkg` is a 508-feature polygon GeoPackage created by joining the microcatchment polygons (`MICRO_ID`) to the long-form LightGBM SHAP table (`basin_id`) for `explained_class_code = 5`. It contains class labels, six predictor values, signed SHAP values and absolute SHAP magnitudes.

The field prefixes `F01`–`F06` correspond to the ranked predictors in `class5_top_six_shap_feature_ranking.csv`. For each prefix, `_value` is the scaled predictor value, `_shap` is the signed SHAP contribution, and `_abs_shap` is its magnitude. These are not interchangeable.

The SHAP values explain the LightGBM surrogate's class-5 output relative to the VGAE-derived target; they are not causal effects or observed flood probabilities. The compact `class5_top_six_predictor_values_508.csv` file contains predictor-value summaries and should not be mistaken for the SHAP contribution table.
