# Local path configuration

The recovered upstream scripts were originally authored against project-specific Windows folders. Before running a stage, update the path constants at the top of that script to point to the local source data and output directories.

The spatial SHAP atlas script is configured to use repository-relative inputs by default:

- `data/spatial_explainability/lightgbm_shap_values_long_all_classes_508.csv`
- `data/spatial_units/krb_microcatchment_boundaries_508.shp`

For the raster stages, provide the original raster stack and basin boundary locally. Raw raster files are intentionally not included in this GitHub bundle.
