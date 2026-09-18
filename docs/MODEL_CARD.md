# Model Card — Day-28 Student Academic Risk Model

## Purpose
Estimate later academic risk for supportive early intervention using information available through Day 28.

## Target
- At risk: final outcome = Fail or Withdrawn
- Not at risk: final outcome = Pass or Distinction

## Intended use
Support prioritization for tutoring, assessment follow-up, engagement outreach, and academic support. Not intended for autonomous admissions, grading, discipline, exclusion, or other high-stakes decisions.

## Data boundary
- Day 0–28 only for predictive behavior/assessment features.
- `final_result` is target-only.
- `date_unregistration` is cohort eligibility only.
- `id_student` is used for joining/group splitting only.

## Validation design
A persisted group-aware split prevents a student from appearing in both development and the locked evaluation holdout. Target-facing EDA, baseline comparison, hyperparameter tuning, calibration, threshold selection, ANN diagnostics, and sensitive-feature ablation are development-only in the current reproducible pipeline.

**Caveat:** earlier iterative development viewed holdout diagnostics. The packaged holdout is therefore a locked evaluation holdout, not a pristine never-seen external test. A future cohort or external institution is required for strict external validation.

## Primary model
Logistic Regression (`C=0.05`, `class_weight=balanced`) selected by development 5-fold group-aware CV F1, with recall and ROC-AUC as tie-breakers.

## Operating performance
At the development-selected operating threshold (~0.3197) on the locked evaluation holdout:
- Recall ≈ 0.8320
- F1 ≈ 0.7011
- ROC-AUC ≈ 0.7977
- PR-AUC ≈ 0.7770
- Brier ≈ 0.1791
- 10-bin quantile ECE ≈ 0.0148

The threshold favors recall, so false-positive support alerts are expected.

## Explainability
- Global SHAP output uses mean |SHAP| as **importance magnitude only**.
- Local signed SHAP explains individual predictions and includes original feature values.
- Logistic Regression coefficients provide transformed-feature direction diagnostics.
- None of these establish causality.

## Recommendations
Recommendations are rule-based and use actionable academic/engagement signals. Demographic attributes are not used by the recommendation engine.

## Sensitive attributes and fairness
The primary prediction model can use demographic/context variables present in OULAD. A subgroup audit and a development-only sensitive-feature ablation are included. Removing `gender`, `age_band`, `disability`, `imd_band`, `highest_education`, and `region` lowers development CV F1 from about **0.6704** to **0.6593** and ROC-AUC from about **0.7850** to **0.7738**. This documents a performance/fairness trade-off; it does not prove either configuration is universally fair.

## Limitations
- OULAD reflects a specific distance-learning context.
- External/future-cohort validation is required before deployment elsewhere.
- Low modeled risk is not guaranteed success.
- The model predicts association with future outcome; it does not diagnose causes.
- Subgroup metrics for small samples are unstable; rows under n=200 are flagged in the audit.
