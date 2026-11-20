STEP 05 — FOUR STANDARD SHAP FIGURES
Model: LightGBM
Target: VGAE_FSM5 (codes 1–5)
Microcatchments explained: 508
Predictors: 34

FIG01: Multiclass beeswarm; five panels, one per class, top 10 predictors.
FIG02: Global bar; macro-average mean absolute SHAP across five classes.
FIG03: Local class-5 waterfall for basin_id 139, node_index 138.
Selection: correctly classified class-5 microcatchment closest to the class-5 feature median.
Actual class: 5; predicted class: 5.
Predicted probability for class 5: 0.991403.
FIG04: Four class-5 dependence plots.

SCIENTIFIC LIMITATIONS
1. SHAP explains the LightGBM surrogate approximation of VGAE-derived
   susceptibility labels, not the internal VGAE model and not observed flood events.
2. SHAP attributions are not causal effects.
3. Waterfall uses LightGBM class-output units. The class-5 probability
   is displayed separately and is not the waterfall endpoint.
4. The input modelling matrix contains scaled predictors. Dependence plots
   therefore show scaled predictor values, not physical units or thresholds.
5. This script explains the existing 508-row dataset and does not refit the
   surrogate. Interpretive fidelity should be reported alongside Step 3
   held-out performance; random holdout does not prove spatial transferability.
6. Macro-average mean |SHAP| is a descriptive equal-weight summary across
   class outputs, not a percentage of total influence.
7. The representative local case is selected by a reproducible rule, not
   chosen because it has an extreme score.
