# Data inventory

- `spatial_units/`: canonical 508-microcatchment boundary shapefile and feature-attribute GeoPackage.
- `flood_susceptibility/`: final VGAE five-class FSM output and multi-layer VGAE/DinkNet comparison output.
- `spatial_explainability/`: Class-5 SHAP spatial vector, long-form SHAP values, top-six factor lookup and predictor-value table.
- `graph/`: graph matrices, directed hydrological edges and node order.
- `graph_construction/`: feature matrix, graph-construction products, node mapping and outlet-routing validation tables.
- `processed/`: source feature and SHAP tables used by the analysis scripts.

All spatial deliverables use the supplied EPSG:32645 projected coordinate reference system. Check individual layer metadata before combining with other data.
