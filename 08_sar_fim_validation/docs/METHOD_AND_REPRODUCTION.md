# Five-event SAR/FIM validation (VGAE-only LOBO)

## Scope
This module evaluates the fixed K=10 VGAE cluster assignments against five
SAR-derived flood-inundation maps (FIMs): 2020, 2021, 2022, 2023 and 2024.
It uses exactly two basin-level observed-flood targets:

- `Flood_fraction_ge_1pct`: observed flood fraction >= 0.01
- `Flood_fraction_ge_5pct`: observed flood fraction >= 0.05

The FIM coding convention is 1 = flood and 0 = non-flood. Explicit NoData
pixels and non-binary values are excluded and reported by the validation script.
The original `VGAE_FSM_cluster10` assignments are read-only and are not
retrained, changed, or reclassified.

## LOBO score
For each event and each basin, the basin is excluded and the pixel-weighted
flood fraction is calculated from all *other* basins in the same original
K=10 cluster. That value is assigned as the excluded basin's LOBO score.
ROC-AUC and average precision use the continuous LOBO score. The threshold-
dependent confusion-matrix metrics use the event-wise median LOBO score,
matching the supplied analysis implementation.

## Interpretation and limitations
This is a retrospective, same-event, peer-cluster association analysis. It is
not an independent prospective forecast, does not establish transfer to an
unseen flood event, and may retain spatial dependence within clusters. The
median-score cutoff is descriptive rather than a calibrated operational
decision threshold. Average precision must be interpreted against each
event/target's positive-class prevalence.

## Running
1. Install dependencies from the repository `requirements.txt`.
2. Edit the `GPKG`, `EVENT_RASTERS`, and `OUTPUT_DIR` paths near the top of
   `scripts/18_vgae_five_event_sar_fim_lobo_validation_1pct_5pct.py`.
3. Run the validation script. It writes basin/event scores, cluster summaries,
   metrics, QC, and an Excel workbook to `OUTPUT_DIR`.
4. Edit `BASE_DIR` and `GPKG` near the top of
   `scripts/19_generate_sar_validation_journal_figures.py` to point to those
   outputs and the source GeoPackage, then run it to regenerate the six
   detailed journal figure groups.

The result tables currently included in `results/tables/` and six generated
figures are the supplied run outputs. The four additional summary figures are
included as Figures 07–10. Figures are preserved as supplied; they are not
recomputed by packaging this repository.
