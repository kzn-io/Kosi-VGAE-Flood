# Surrogate modelling and SHAP

Run in sequence from the repository root (or invoke each script by its path):

1. `12_validate_surrogate_model_inputs.py` — validate feature, label and geometry alignment.
2. `13_create_stratified_model_validation_split.py` — create a reproducible stratified 80:20 split.
3. `14_benchmark_multiclass_surrogate_models.py` — compare LightGBM, Random Forest and XGBoost surrogates.
4. `15_calculate_multiclass_shap_explanations.py` — calculate class-wise SHAP for all three surrogates.
5. `16_export_publication_quality_shap_figures.py` — create selected LightGBM global, local and dependence figures.

The separate spatial mapping stage is `../07_spatial_explainability/17_generate_class5_spatial_shap_atlas.py`.

All figure scripts export 600-dpi PNG and vector PDF. Titles are concise; full predictor names and methods are retained in tables and reports.

**Interpretation:** SHAP explains surrogate outputs that approximate `VGAE_FSM5`. It is not a causal analysis and is not direct observed flood-occurrence prediction. The random stratified holdout is not spatially independent validation.
