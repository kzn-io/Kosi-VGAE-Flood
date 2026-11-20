# Source Code Index

All Python scripts included in this repository use sequential stage identifiers and descriptive filenames.

| ID | Script | Purpose |
|---:|---|---|
| 01 | `01_align_environmental_rasters_to_reference_grid.py` | Harmonise continuous/categorical predictor rasters to the selected reference grid. |
| 02 | `02_derive_terrain_and_hydrological_predictors.py` | Reconstruct terrain and hydrological predictor layers. |
| 03 | `03_construct_microcatchment_predictor_features.py` | Construct microcatchment-level predictor features from available source layers. |
| 04 | `04_audit_flood_susceptibility_model_inputs.py` | Audit the available FSM input data and project artefacts. |
| 05 | `05_audit_vgae_embeddings_and_fsm_inputs.py` | Inspect VGAE embeddings and unsupervised FSM inputs. |
| 06 | `06_validate_feature_geometry_and_graph_alignment.py` | Check alignment among feature rows, polygons, node mapping and graph inputs. |
| 07 | `07_construct_microcatchment_spatial_adjacency.py` | Construct undirected spatial adjacency between microcatchments. |
| 08 | `08_build_508_node_spatial_graph.py` | Build/export the 508-node spatial graph. |
| 09 | `09_generate_vgae_five_class_susceptibility.py` | Generate/export the available VGAE five-class FSM product. |
| 10 | `10_run_unsupervised_susceptibility_regionalization.py` | Run downstream unsupervised susceptibility/regionalization analysis using existing model artefacts. |
| 11 | `11_evaluate_embedding_clusters.py` | Evaluate embedding cluster solutions. |
| 12 | `12_validate_surrogate_model_inputs.py` | Validate the surrogate-model input table and labels. |
| 13 | `13_create_stratified_model_validation_split.py` | Create the stratified hold-out split. |
| 14 | `14_benchmark_multiclass_surrogate_models.py` | Benchmark LightGBM, Random Forest and XGBoost surrogate classifiers. |
| 15 | `15_calculate_multiclass_shap_explanations.py` | Calculate multiclass SHAP explanations for the surrogate models. |
| 16 | `16_export_publication_quality_shap_figures.py` | Export global, class-specific and local SHAP figures. |
| 17 | `17_generate_class5_spatial_shap_atlas.py` | Map signed and absolute Class-5 SHAP contributions. |

## Interpretation and reproducibility boundary

The filenames have been standardised; the scientific methods and model outputs have not been silently changed by renaming. Review each script's path configuration before running it. The recovered source set does not include the exact historical raw-zonal-statistics-to-scaled-feature transformation or original VGAE training implementation; see `docs/REPRODUCIBILITY_GAPS.md`. The available unsupervised stage consumes existing model artefacts and must not be represented as a fresh VGAE training script.

## SAR/FIM validation additions
- `08_sar_fim_validation/scripts/18_vgae_five_event_sar_fim_lobo_validation_1pct_5pct.py` — five-event LOBO validation for >=1% and >=5% flood fractions.
- `08_sar_fim_validation/scripts/19_generate_sar_validation_journal_figures.py` — journal figure generation from the validation outputs.
