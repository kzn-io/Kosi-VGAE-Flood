# SHAP figure standards

- Raster exports: 600 dpi PNG.
- Vector exports: PDF (preferred for journal submission and final layout).
- Figure headings and axis labels are intentionally concise.
- Feature names are retained in full in accompanying CSV tables.
- Predictor values are scaled values from the supplied modelling matrix.
- SHAP values explain the LightGBM surrogate of VGAE-derived classes; they are not causal effects or observed flood probabilities.
- The 80:20 random stratified holdout is not spatially independent validation.
