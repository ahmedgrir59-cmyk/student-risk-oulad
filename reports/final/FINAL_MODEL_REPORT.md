# Final Model Report

## Selected model
- Model: **logistic_regression**
- Parameters: `{"C": 0.05, "class_weight": "balanced"}`
- Selection: highest mean 5-fold group-aware CV F1 on development data.
- CV F1: **0.6704**
- CV Recall: **0.6580**
- CV ROC-AUC: **0.7850**

## Locked evaluation holdout
- Rows: **5,504**
- Student overlap with development: **0**
- Holdout labels are not used by the current reproducible tuning, calibration, threshold-selection, or automated model-selection pipeline.
- Methodology caveat: during earlier iterative project development, holdout diagnostics were viewed. Therefore this is described as a **locked evaluation holdout**, not a pristine never-seen external test. A future cohort/external dataset is recommended for strict external validation.

## Holdout metrics (development-tuned threshold = 0.3197)
- Accuracy: **0.6871**
- Precision: **0.6059**
- Recall: **0.8320**
- F1: **0.7011**
- ROC-AUC: **0.7977**
- PR-AUC: **0.7770**
- Brier score: **0.1791**
- 10-bin quantile ECE: **0.0148**
- Confusion matrix: TN=1762, FP=1314, FN=408, TP=2020

## Risk levels
- Low: probability < **0.2509**
- Medium: **0.2509** <= probability < **0.3197**
- High: probability >= **0.3197**
- Low means lower relative modeled risk, **not guaranteed success**.

## Explainability and interventions
- `shap_global_importance.csv` reports **global importance magnitude** using mean |SHAP|; it intentionally does not label features globally as risk-increasing/protective.
- `model_coefficients.csv` reports transformed Logistic Regression coefficient signs as model-direction diagnostics, with explicit non-causal warnings.
- `shap_local_top_factors_sample.csv` reports signed local SHAP contributors with source feature and original value for individual predictions.
- Recommendations use only actionable academic/engagement features, not sensitive demographic attributes.
- The subgroup audit is reporting-only and does not alter student interventions.

## Interpretation warning
SHAP and coefficients explain model behavior/association. They do not prove causal effects. This system is intended as a support/early-warning aid, not an autonomous grading, admissions, discipline, or exclusion system.
