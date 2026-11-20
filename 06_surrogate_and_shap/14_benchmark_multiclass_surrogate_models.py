# ================================================================
# KRB VGAE-SHAP | STEP 3: TRAIN AND COMPARE SURROGATE CLASSIFIERS
# REVISED: XGBoost target-label encoding corrected (0..4 internally).
# ================================================================
# Run after STEP 2 has completed with STATUS: PASS.
#
# Models:
#   1. LightGBM multiclass
#   2. Random Forest multiclass
#   3. XGBoost multiclass
#
# Evaluation:
#   - Five-fold stratified CV on the training set
#   - One final evaluation on the untouched 20% hold-out test set
#   - Accuracy, balanced accuracy, macro-F1, weighted-F1
#   - Confusion matrices and per-class precision/recall/F1
#
# These are surrogate models that reproduce VGAE_FSM5 labels.
# They are NOT direct flood-occurrence prediction models.
# ================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ------------------------- PATHS ----------------------------------
ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "results" / "shap_analysis"
STEP2_DIR = BASE_DIR / "step02_split"
TRAIN_FILE = STEP2_DIR / "train_80pct.csv"
TEST_FILE = STEP2_DIR / "test_20pct.csv"
FEATURES_FILE = STEP2_DIR / "predictors.json"
STEP2_REPORT = STEP2_DIR / "split_report.txt"
OUTPUT_DIR = BASE_DIR / "step03_surrogate_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_CSV = OUTPUT_DIR / "model_metrics.csv"
CV_CSV = OUTPUT_DIR / "cross_validation.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"
REPORT_TXT = OUTPUT_DIR / "model_report.txt"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- SETTINGS -------------------------------

TARGET = "VGAE_FSM5"
ID_COLUMNS = ["basin_id", "node_index"]
CLASS_LABELS = [1, 2, 3, 4, 5]
RANDOM_STATE = 42
N_SPLITS = 5

# ------------------------- INPUT VALIDATION -----------------------

for p in [TRAIN_FILE, TEST_FILE, FEATURES_FILE, STEP2_REPORT]:
    if not p.exists():
        raise FileNotFoundError(f"Required Step 2 output missing: {p}")

if "STATUS: PASS" not in STEP2_REPORT.read_text(encoding="utf-8"):
    raise RuntimeError("Step 2 report is not marked STATUS: PASS.")

config = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
predictors = config["predictors"]

if len(predictors) != 34:
    raise ValueError(f"Expected 34 predictors; found {len(predictors)}.")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

for name, data in [("train", train), ("test", test)]:
    needed = set(ID_COLUMNS + [TARGET] + predictors)
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError(f"{name} data missing columns: {sorted(missing)}")
    if data[list(needed)].isna().any().any():
        raise ValueError(f"{name} data contains missing values.")
    if not np.isfinite(data[predictors].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} predictors contain infinite values.")
    if sorted(data[TARGET].astype(int).unique().tolist()) != CLASS_LABELS:
        raise ValueError(f"{name} does not contain all five classes.")

if set(train["basin_id"]).intersection(set(test["basin_id"])):
    raise ValueError("Train/test basin overlap detected.")

X_train = train[predictors].astype(float)
y_train = train[TARGET].astype(int)
X_test = test[predictors].astype(float)
y_test = test[TARGET].astype(int)

# ------------------------- IMPORT OPTIONAL LIBRARIES --------------

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise ImportError(
        "LightGBM is not installed. Run: pip install lightgbm"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    ) from exc

# ------------------------- DEFINE MODELS --------------------------

models = {
    "LightGBM": LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.10,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1
    ),

    "Random_Forest": RandomForestClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=2,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.10,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# ------------------------- CROSS-VALIDATION -----------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted"
}

cv_rows = []
metric_rows = []
prediction_output = test[ID_COLUMNS + [TARGET]].copy()

report_lines = [
    "KRB VGAE-SHAP | STEP 3 SURROGATE MODEL REPORT",
    "=" * 70,
    f"Training observations: {len(train)}",
    f"Testing observations: {len(test)}",
    f"Predictors: {len(predictors)}",
    f"CV: {N_SPLITS}-fold StratifiedKFold on training data",
    "Test set: untouched until final evaluation",
    ""
]

for model_name, model in models.items():
    print(f"\n{'=' * 65}\nTraining {model_name}\n{'=' * 65}")

    # XGBoost's sklearn interface requires contiguous class labels 0..4.
    # The source VGAE target remains 1..5; encode only for XGBoost fitting
    # and decode its predictions back to the original 1..5 class codes.
    if model_name == "XGBoost":
        y_train_model = y_train - 1
    else:
        y_train_model = y_train

    cv_result = cross_validate(
        model,
        X_train,
        y_train_model,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
        error_score="raise"
    )

    cv_summary = {}
    for metric_name in scoring:
        values = cv_result[f"test_{metric_name}"]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1))
        cv_summary[metric_name] = (mean_value, std_value)

        cv_rows.append({
            "Model": model_name,
            "Metric": metric_name,
            "CV_Mean": mean_value,
            "CV_SD": std_value,
            "Fold_Values": ";".join(f"{v:.6f}" for v in values)
        })

    # Fit final model on all training observations.
    model.fit(X_train, y_train_model)

    # Evaluate exactly once on the held-out test set.
    y_pred = model.predict(X_test).astype(int)
    if model_name == "XGBoost":
        y_pred = y_pred + 1  # Restore original VGAE class codes: 1..5

    metrics = {
        "Model": model_name,
        "Test_Accuracy": accuracy_score(y_test, y_pred),
        "Test_Balanced_Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Test_Macro_F1": f1_score(y_test, y_pred, average="macro"),
        "Test_Weighted_F1": f1_score(y_test, y_pred, average="weighted"),
        "Test_Macro_Precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Test_Macro_Recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "CV_Macro_F1_Mean": cv_summary["f1_macro"][0],
        "CV_Macro_F1_SD": cv_summary["f1_macro"][1],
        "CV_Balanced_Accuracy_Mean": cv_summary["balanced_accuracy"][0],
        "CV_Balanced_Accuracy_SD": cv_summary["balanced_accuracy"][1]
    }
    metric_rows.append(metrics)

    prediction_output[f"{model_name}_prediction"] = y_pred

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in CLASS_LABELS],
        columns=[f"Predicted_{c}" for c in CLASS_LABELS]
    )
    cm_df.to_csv(
        OUTPUT_DIR / f"KRB_SHAP_STEP03_{model_name}_CONFUSION_MATRIX.csv"
    )

    class_report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(class_report).T.to_csv(
        OUTPUT_DIR / f"KRB_SHAP_STEP03_{model_name}_CLASS_REPORT.csv"
    )

    joblib.dump(
        model,
        MODEL_DIR / f"KRB_SHAP_{model_name}_SURROGATE.joblib"
    )

    report_lines.extend([
        f"MODEL: {model_name}",
        "-" * 50,
        "Cross-validation (mean ± SD):",
        *[
            f"{key}: {mean:.4f} ± {sd:.4f}"
            for key, (mean, sd) in cv_summary.items()
        ],
        "Held-out test metrics:",
        *[
            f"{key}: {value:.4f}"
            for key, value in metrics.items()
            if key.startswith("Test_")
        ],
        "Test confusion matrix (rows=actual, columns=predicted):",
        str(cm),
        "Class-wise report:",
        classification_report(
            y_test, y_pred, labels=CLASS_LABELS, zero_division=0
        ),
        ""
    ])

    print("Test macro-F1:", round(metrics["Test_Macro_F1"], 4))
    print("Test balanced accuracy:", round(metrics["Test_Balanced_Accuracy"], 4))

# ------------------------- SAVE RESULTS ---------------------------

metrics_df = pd.DataFrame(metric_rows)
cv_df = pd.DataFrame(cv_rows)

metrics_df.to_csv(METRICS_CSV, index=False)
cv_df.to_csv(CV_CSV, index=False)
prediction_output.to_csv(PREDICTIONS_CSV, index=False)

# Rank only as a descriptive comparison; do not automatically select
# a model solely by test score. Prefer CV stability + held-out fidelity.
ranked = metrics_df.sort_values(
    ["CV_Macro_F1_Mean", "Test_Macro_F1"],
    ascending=False
)

report_lines.extend([
    "MODEL COMPARISON (descriptive ordering by CV macro-F1):",
    ranked[[
        "Model", "CV_Macro_F1_Mean", "CV_Macro_F1_SD",
        "Test_Macro_F1", "Test_Balanced_Accuracy"
    ]].to_string(index=False),
    "",
    "INTERPRETATION:",
    "These models approximate the VGAE-derived five-class labels.",
    "Their scores measure surrogate fidelity, not flood prediction accuracy.",
    "The random hold-out is not a spatially independent validation design.",
    "",
    "OUTPUTS:",
    str(METRICS_CSV),
    str(CV_CSV),
    str(PREDICTIONS_CSV),
    str(MODEL_DIR),
    "",
    "STATUS: PASS"
])

REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")

print("\n" + "=" * 70)
print("STEP 3 COMPLETED")
print("=" * 70)
print(metrics_df.to_string(index=False))
print("\nSaved results to:", OUTPUT_DIR)
print("Review CV stability and held-out fidelity before selecting a model.")
