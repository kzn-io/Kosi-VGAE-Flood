# Microcatchment Features and Graph-Construction Data

This directory preserves the supplied spatial-unit, node-feature, graph-topology and outlet-routing validation products for the 508-microcatchment Kosi River Basin workflow.

## Directory structure

- `spatial_units/`
  - The canonical polygon boundary is maintained once at `data/spatial_units/krb_microcatchment_boundaries_508.*`.
  - `krb_microcatchment_features_508.gpkg`: GeoPackage containing the 508 microcatchments and their feature attributes, including `basin_id`.
- `node_features/`
  - `microcatchment_feature_matrix_508.csv`: 508-row feature matrix; `basin_id` is the node-to-basin identifier.
- `topology/`
  - `candidate_spatial_adjacency_edges.csv`: candidate adjacency edge list (`source_id`, `target_id`).
  - `directed_hydrological_edges.csv`: directed hydrological edge list with `crossing_pixels`.
  - `graph_node_order_508.csv`: explicit mapping between zero-based `node_index` and one-based `basin_id`.
  - `candidate_adjacency_matrix_508.npz`: sparse 508 × 508 candidate adjacency matrix in CSR representation.
  - `directed_hydrological_adjacency_508.npz`: sparse 508 × 508 directed adjacency matrix in CSR representation.
- `outlet_routing_validation/`
  - `outlet_trace_edge_diagnostics.csv`: edge-level outlet/crossing-pixel diagnostics.
  - `microcatchment_outlet_summary.csv`: per-microcatchment candidate outlet summary.

## Identifiers and dimensions

- Number of microcatchments / graph nodes: 508.
- `basin_id`: 1–508.
- `node_index`: 0–507; supplied mapping follows `basin_id = node_index + 1`.
- Spatial reference of the supplied polygon layer: EPSG:32645.
- Feature matrix: 508 rows × 38 columns (`basin_id` plus 37 predictor summaries/fractions).
- Candidate adjacency matrix: 508 × 508 CSR; 2,930 stored non-zero entries.
- Directed hydrological adjacency matrix: 508 × 508 CSR; 2,846 stored non-zero entries.

Candidate spatial adjacency and directed hydrological products represent distinct graph relationships. Keep their edge direction and semantics separate during analysis. Outlet-routing validation tables are retained as supplied and should be interpreted alongside the graph-construction methodology.

## Consistency check

The GeoPackage attribute table and CSV feature matrix contain the same 508 `basin_id` records and the same 38 attribute fields. Their numeric values agree to floating-point precision (maximum absolute difference approximately 1.1 × 10⁻¹⁶). This validates the supplied tabular correspondence; it does not independently validate the scientific assumptions used to derive each feature or edge.
