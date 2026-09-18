from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from recommendation_engine import generate_recommendations
from split_utils import CV_SEED, DROP_FROM_X, GROUP, TARGET, indices_from_manifest, load_or_create_manifest


def raw_score(model, X):
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X), dtype=float)
    p = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def choose_thresholds(y, prob):
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_i = int(np.nanargmax(f1))
    high = float(thresholds[best_i])

    valid_mask = recall[:-1] >= 0.90
    low = float(np.max(thresholds[valid_mask])) if np.any(valid_mask) else float(np.quantile(prob, 0.25))
    if low >= high:
        low = max(0.0, high * 0.60)
    return low, high, float(f1[best_i])


def risk_level(p, low, high):
    if p >= high:
        return 'High'
    if p >= low:
        return 'Medium'
    return 'Low'


def expected_calibration_error(y, prob, n_bins=10):
    """Quantile-bin ECE for a compact calibration diagnostic."""
    frame = pd.DataFrame({'y': np.asarray(y, dtype=int), 'p': np.asarray(prob, dtype=float)})
    # duplicates='drop' makes this robust to repeated probabilities.
    frame['bin'] = pd.qcut(frame['p'], q=n_bins, duplicates='drop')
    grouped = frame.groupby('bin', observed=True).agg(n=('y', 'size'), observed=('y', 'mean'), predicted=('p', 'mean'))
    return float(((grouped['n'] / len(frame)) * (grouped['observed'] - grouped['predicted']).abs()).sum())


def metrics_at_threshold(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, prob)),
        'pr_auc': float(average_precision_score(y, prob)),
        'brier': float(brier_score_loss(y, prob)),
        'ece_10_quantile': expected_calibration_error(y, prob, n_bins=10),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def build_recommendation_config(X_dev: pd.DataFrame) -> dict:
    score = X_dev['assessment_mean_score_28d'].dropna()
    comp = X_dev['assessment_completion_rate_28d'].dropna()
    return {
        'assessment_score_low': float(score.quantile(0.25)) if len(score) else 50.0,
        'completion_rate_low': float(comp.quantile(0.25)) if len(comp) else 0.75,
        'days_since_activity_high': float(X_dev['days_since_last_activity_28d'].dropna().quantile(0.75)),
        'active_days_last7d_low': float(X_dev['active_days_last7d'].quantile(0.25)),
        'clicks_last7d_low': float(X_dev['clicks_last7d'].quantile(0.25)),
        'weekly_click_slope_low': float(min(0.0, X_dev['weekly_click_slope_28d'].quantile(0.25))),
    }


def fairness_table(raw_holdout: pd.DataFrame, y, prob, threshold) -> pd.DataFrame:
    pred = (prob >= threshold).astype(int)
    base = raw_holdout.copy()
    base['_y'] = np.asarray(y)
    base['_pred'] = pred
    rows = []
    for attr in ['gender', 'age_band', 'disability', 'imd_band', 'highest_education']:
        if attr not in base:
            continue
        for value, d in base.groupby(attr, dropna=False):
            if len(d) < 50:
                continue
            yt = d['_y'].to_numpy()
            yp = d['_pred'].to_numpy()
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            rows.append({
                'attribute': attr,
                'group': str(value),
                'n': int(len(d)),
                'small_group_warning': bool(len(d) < 200),
                'actual_risk_rate': float(yt.mean()),
                'predicted_risk_rate': float(yp.mean()),
                'precision': float(precision_score(yt, yp, zero_division=0)),
                'recall': float(recall_score(yt, yp, zero_division=0)),
                'false_positive_rate': float(fp / (fp + tn)) if fp + tn else np.nan,
                'accuracy': float(accuracy_score(yt, yp)),
            })
    return pd.DataFrame(rows)


def source_feature_name(transformed_feature: str, raw_columns: list[str]) -> str:
    if transformed_feature.startswith('num__'):
        return transformed_feature[len('num__'):]
    if transformed_feature.startswith('cat__'):
        tail = transformed_feature[len('cat__'):]
        for col in sorted(raw_columns, key=len, reverse=True):
            if tail == col or tail.startswith(col + '_'):
                return col
    return transformed_feature


def original_value_for_feature(transformed_feature: str, row: pd.Series, raw_columns: list[str]):
    src = source_feature_name(transformed_feature, raw_columns)
    if src in row.index:
        val = row[src]
        return None if pd.isna(val) else val
    return None


def main(data: Path, tuning_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Remove legacy scientifically ambiguous directional SHAP exports if present.
    for legacy in ['top_risk_increasing_factors.csv', 'top_protective_factors.csv']:
        p = output_dir / legacy
        if p.exists():
            p.unlink()

    df = pd.read_csv(data)
    X = df.drop(columns=DROP_FROM_X)
    y = df[TARGET].astype(int)
    groups = df[GROUP]
    manifest = load_or_create_manifest(df, tuning_dir / 'locked_split_manifest.csv')
    dev_idx, holdout_idx = indices_from_manifest(manifest)
    X_dev, y_dev, g_dev = X.iloc[dev_idx].copy(), y.iloc[dev_idx].copy(), groups.iloc[dev_idx].copy()
    X_holdout, y_holdout = X.iloc[holdout_idx].copy(), y.iloc[holdout_idx].copy()
    raw_holdout = df.iloc[holdout_idx].copy()

    print('Stage 1: load development-selected model', flush=True)
    selection = json.loads((tuning_dir / 'model_selection.json').read_text(encoding='utf-8'))
    name = selection['selected_model']
    base = joblib.load(tuning_dir / 'candidate_models' / f'{name}_tuned_dev.joblib')

    print('Stage 2: development-only OOF calibration scores', flush=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    oof_score = np.full(len(X_dev), np.nan)
    for fold, (tr, va) in enumerate(cv.split(X_dev, y_dev, g_dev), 1):
        print(f'  calibration fold {fold}/5', flush=True)
        m = clone(base)
        m.fit(X_dev.iloc[tr], y_dev.iloc[tr])
        oof_score[va] = raw_score(m, X_dev.iloc[va])
    assert np.isfinite(oof_score).all()

    print('Stage 3: fit calibrator and thresholds on development OOF only', flush=True)
    calibrator = LogisticRegression(solver='lbfgs', random_state=CV_SEED)
    calibrator.fit(oof_score.reshape(-1, 1), y_dev)
    oof_prob = calibrator.predict_proba(oof_score.reshape(-1, 1))[:, 1]
    low_thr, high_thr, oof_best_f1 = choose_thresholds(y_dev.to_numpy(), oof_prob)

    print('Stage 4: fit final base model on all development rows, then evaluate locked holdout', flush=True)
    base.fit(X_dev, y_dev)
    holdout_raw = raw_score(base, X_holdout)
    holdout_prob = calibrator.predict_proba(holdout_raw.reshape(-1, 1))[:, 1]

    final_metrics = metrics_at_threshold(y_holdout.to_numpy(), holdout_prob, high_thr)
    default_metrics = metrics_at_threshold(y_holdout.to_numpy(), holdout_prob, 0.5)
    calibration = {
        'oof_brier_calibrated': float(brier_score_loss(y_dev, oof_prob)),
        'oof_ece_10_quantile': expected_calibration_error(y_dev, oof_prob),
        'final_holdout_brier_calibrated': float(brier_score_loss(y_holdout, holdout_prob)),
        'final_holdout_ece_10_quantile': expected_calibration_error(y_holdout, holdout_prob),
    }

    threshold_info = {
        'low_risk_upper_threshold': low_thr,
        'high_risk_lower_threshold': high_thr,
        'medium_range': f'[{low_thr:.6f}, {high_thr:.6f})',
        'threshold_method': 'Low boundary is the highest OOF-calibrated probability threshold maintaining >=90% recall. High/decision boundary maximizes development OOF F1.',
        'oof_f1_at_high_threshold': oof_best_f1,
    }
    (output_dir / 'risk_thresholds.json').write_text(json.dumps(threshold_info, indent=2), encoding='utf-8')
    (output_dir / 'final_test_metrics.json').write_text(json.dumps({
        'selected_model': name,
        'selected_params': selection['selected_params'],
        'tuned_threshold_metrics': final_metrics,
        'threshold_0_5_metrics': default_metrics,
        'calibration': calibration,
        'holdout_methodology_note': selection.get('holdout_methodology_note', ''),
    }, indent=2), encoding='utf-8')

    cfg = build_recommendation_config(X_dev)
    (output_dir / 'recommendation_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    print('Stage 5: holdout predictions, risk bands, error and subgroup audit', flush=True)
    pred = (holdout_prob >= high_thr).astype(int)
    levels = [risk_level(float(p), low_thr, high_thr) for p in holdout_prob]
    rec_strings = []
    for (_, row), level in zip(raw_holdout.iterrows(), levels):
        if level == 'Low':
            recs = [{
                'factor': 'monitor',
                'recommendation': 'Continue routine monitoring and reinforce current study habits.',
                'reason': 'Calibrated risk is below the low-risk boundary; low risk is relative and does not guarantee success.',
            }]
        else:
            recs = generate_recommendations(row, cfg)
        rec_strings.append(json.dumps(recs, ensure_ascii=False))

    out = pd.DataFrame({
        'row_index': holdout_idx,
        'id_student': raw_holdout['id_student'].to_numpy(),
        'code_module': raw_holdout['code_module'].to_numpy(),
        'code_presentation': raw_holdout['code_presentation'].to_numpy(),
        'actual_at_risk': y_holdout.to_numpy(),
        'predicted_at_risk': pred,
        'risk_probability': holdout_prob,
        'risk_level': levels,
        'recommendations_json': rec_strings,
    })
    out.to_csv(output_dir / 'final_holdout_predictions.csv', index=False)
    level_dist = out.groupby('risk_level').agg(
        n=('risk_level', 'size'),
        actual_risk_rate=('actual_at_risk', 'mean'),
        mean_probability=('risk_probability', 'mean'),
    ).reset_index()
    (output_dir / 'risk_level_distribution.json').write_text(level_dist.to_json(orient='records', indent=2), encoding='utf-8')

    err = raw_holdout[['id_student', 'code_module', 'code_presentation', 'active_days_last7d', 'clicks_last7d', 'days_since_last_activity_28d', 'assessment_mean_score_28d', 'assessments_missed_28d', 'assessment_completion_rate_28d']].copy()
    err['actual'] = y_holdout.to_numpy()
    err['predicted'] = pred
    err['risk_probability'] = holdout_prob
    err['error_type'] = np.select(
        [(err.actual == 1) & (err.predicted == 0), (err.actual == 0) & (err.predicted == 1)],
        ['False Negative', 'False Positive'],
        default='Correct',
    )
    err[err.error_type != 'Correct'].sort_values(['error_type', 'risk_probability']).to_csv(output_dir / 'error_analysis_cases.csv', index=False)
    err.groupby('error_type').size().rename('count').reset_index().to_csv(output_dir / 'error_analysis_summary.csv', index=False)
    fairness_table(raw_holdout, y_holdout.to_numpy(), holdout_prob, high_thr).to_csv(output_dir / 'fairness_support_audit.csv', index=False)

    print('Stage 6: explainability', flush=True)
    prep = base.named_steps['prep']
    est = base.named_steps['model']
    feature_names = np.asarray(prep.get_feature_names_out(), dtype=str)
    raw_feature_columns = X.columns.tolist()

    # Global SHAP importance is computed on a development sample so the holdout is not
    # needed for global interpretability. Mean |SHAP| measures importance only; it is not
    # labeled as globally risk-increasing/protective.
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_dev), size=min(500, len(X_dev)), replace=False)
    explain_dev_idx = rng.choice(len(X_dev), size=min(1500, len(X_dev)), replace=False)
    X_bg = prep.transform(X_dev.iloc[bg_idx])
    X_explain_dev = prep.transform(X_dev.iloc[explain_dev_idx])
    explainer = shap.LinearExplainer(est, X_bg)
    sv_dev = explainer(X_explain_dev)
    mean_abs = np.abs(sv_dev.values).mean(axis=0)
    shap_global = pd.DataFrame({
        'feature': feature_names,
        'source_feature': [source_feature_name(f, raw_feature_columns) for f in feature_names],
        'mean_abs_shap': mean_abs,
        'interpretation': 'Global importance magnitude only; larger values mean greater average influence on model output. No causal or global directional claim.',
    }).sort_values('mean_abs_shap', ascending=False)
    shap_global.to_csv(output_dir / 'shap_global_importance.csv', index=False)

    # For a linear Logistic Regression model, coefficient sign is a more defensible
    # description of transformed-feature direction than mean signed SHAP. It is still an
    # association in model log-odds, not a causal effect.
    if hasattr(est, 'coef_'):
        coef = np.asarray(est.coef_).reshape(-1)
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'source_feature': [source_feature_name(f, raw_feature_columns) for f in feature_names],
            'coefficient': coef,
            'abs_coefficient': np.abs(coef),
            'model_direction': np.where(coef > 0, 'higher model log-odds of risk when this transformed feature increases/is active', 'lower model log-odds of risk when this transformed feature increases/is active'),
            'interpretation_warning': 'Coefficient direction is conditional on the fitted transformed feature space and regularization; it is not causal.',
        }).sort_values('abs_coefficient', ascending=False)
        coef_df.to_csv(output_dir / 'model_coefficients.csv', index=False)

    top = shap_global.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top['feature'], top['mean_abs_shap'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 20 Global Model Features by SHAP Importance')
    plt.tight_layout()
    plt.savefig(fig_dir / 'shap_top20.png', dpi=160)
    plt.close()

    # Local holdout explanations: signed SHAP is valid per prediction. Include source
    # feature and original value so the output is interpretable outside sklearn naming.
    local_explain_idx = rng.choice(len(X_holdout), size=min(800, len(X_holdout)), replace=False)
    X_local = prep.transform(X_holdout.iloc[local_explain_idx])
    X_local_dense = X_local.toarray() if hasattr(X_local, 'toarray') else np.asarray(X_local)
    sv_local = explainer(X_local)
    local_rows = []
    for pos, src_pos in enumerate(local_explain_idx):
        vals = np.asarray(sv_local.values[pos])
        row = X_holdout.iloc[src_pos]
        top_risk = np.where(vals > 0)[0]
        top_risk = top_risk[np.argsort(vals[top_risk])[::-1]][:5] if len(top_risk) else []
        top_protective = np.where(vals < 0)[0]
        top_protective = top_protective[np.argsort(vals[top_protective])][:5] if len(top_protective) else []

        def pack(indices):
            items = []
            for j in indices:
                f = str(feature_names[j])
                src = source_feature_name(f, raw_feature_columns)
                val = original_value_for_feature(f, row, raw_feature_columns)
                if isinstance(val, np.generic):
                    val = val.item()
                items.append({
                    'transformed_feature': f,
                    'source_feature': src,
                    'original_value': val,
                    'transformed_value': float(X_local_dense[pos, j]),
                    'shap_value': float(vals[j]),
                })
            return items

        local_rows.append({
            'holdout_position': int(src_pos),
            'row_index': int(holdout_idx[src_pos]),
            'id_student': int(raw_holdout.iloc[src_pos]['id_student']),
            'risk_probability': float(holdout_prob[src_pos]),
            'top_local_risk_contributors_json': json.dumps(pack(top_risk), ensure_ascii=False),
            'top_local_protective_contributors_json': json.dumps(pack(top_protective), ensure_ascii=False),
            'interpretation_warning': 'Local SHAP explains this prediction; it does not establish causality.',
        })
    pd.DataFrame(local_rows).to_csv(output_dir / 'shap_local_top_factors_sample.csv', index=False)

    print('Stage 7: evaluation figures', flush=True)
    cm = confusion_matrix(y_holdout, pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.xticks([0, 1], ['Not risk', 'At risk'])
    plt.yticks([0, 1], ['Not risk', 'At risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Locked Evaluation Holdout Confusion Matrix')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(fig_dir / 'confusion_matrix.png', dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_holdout, holdout_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_holdout, holdout_prob):.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Locked Evaluation Holdout ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'roc_curve.png', dpi=160)
    plt.close()

    pr, rc, _ = precision_recall_curve(y_holdout, holdout_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rc, pr, label=f"PR-AUC={average_precision_score(y_holdout, holdout_prob):.3f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Locked Evaluation Holdout Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'precision_recall_curve.png', dpi=160)
    plt.close()

    frac, mean = calibration_curve(y_holdout, holdout_prob, n_bins=10, strategy='quantile')
    plt.figure(figsize=(6, 5))
    plt.plot(mean, frac, marker='o')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed risk frequency')
    plt.title('Locked Evaluation Holdout Calibration')
    plt.tight_layout()
    plt.savefig(fig_dir / 'calibration_curve.png', dpi=160)
    plt.close()

    print('Stage 8: persist inference bundle/report', flush=True)
    bundle = {
        'base_model': base,
        'calibrator': calibrator,
        'low_threshold': low_thr,
        'high_threshold': high_thr,
        'recommendation_config': cfg,
        'feature_columns': X.columns.tolist(),
        'selected_model': name,
        'selected_params': selection['selected_params'],
        'prediction_point_day': 28,
        'risk_level_note': 'Low/Medium/High are relative modeled risk bands, not guarantees or diagnoses.',
    }
    joblib.dump(bundle, output_dir / 'final_risk_bundle.joblib')

    report = f"""# Final Model Report

## Selected model
- Model: **{name}**
- Parameters: `{json.dumps(selection['selected_params'])}`
- Selection: highest mean 5-fold group-aware CV F1 on development data.
- CV F1: **{selection['selected_cv_f1']:.4f}**
- CV Recall: **{selection['selected_cv_recall']:.4f}**
- CV ROC-AUC: **{selection['selected_cv_roc_auc']:.4f}**

## Locked evaluation holdout
- Rows: **{len(holdout_idx):,}**
- Student overlap with development: **0**
- Holdout labels are not used by the current reproducible tuning, calibration, threshold-selection, or automated model-selection pipeline.
- Methodology caveat: during earlier iterative project development, holdout diagnostics were viewed. Therefore this is described as a **locked evaluation holdout**, not a pristine never-seen external test. A future cohort/external dataset is recommended for strict external validation.

## Holdout metrics (development-tuned threshold = {high_thr:.4f})
- Accuracy: **{final_metrics['accuracy']:.4f}**
- Precision: **{final_metrics['precision']:.4f}**
- Recall: **{final_metrics['recall']:.4f}**
- F1: **{final_metrics['f1']:.4f}**
- ROC-AUC: **{final_metrics['roc_auc']:.4f}**
- PR-AUC: **{final_metrics['pr_auc']:.4f}**
- Brier score: **{final_metrics['brier']:.4f}**
- 10-bin quantile ECE: **{final_metrics['ece_10_quantile']:.4f}**
- Confusion matrix: TN={final_metrics['tn']}, FP={final_metrics['fp']}, FN={final_metrics['fn']}, TP={final_metrics['tp']}

## Risk levels
- Low: probability < **{low_thr:.4f}**
- Medium: **{low_thr:.4f}** <= probability < **{high_thr:.4f}**
- High: probability >= **{high_thr:.4f}**
- Low means lower relative modeled risk, **not guaranteed success**.

## Explainability and interventions
- `shap_global_importance.csv` reports **global importance magnitude** using mean |SHAP|; it intentionally does not label features globally as risk-increasing/protective.
- `model_coefficients.csv` reports transformed Logistic Regression coefficient signs as model-direction diagnostics, with explicit non-causal warnings.
- `shap_local_top_factors_sample.csv` reports signed local SHAP contributors with source feature and original value for individual predictions.
- Recommendations use only actionable academic/engagement features, not sensitive demographic attributes.
- The subgroup audit is reporting-only and does not alter student interventions.

## Interpretation warning
SHAP and coefficients explain model behavior/association. They do not prove causal effects. This system is intended as a support/early-warning aid, not an autonomous grading, admissions, discipline, or exclusion system.
"""
    (output_dir / 'FINAL_MODEL_REPORT.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--tuning-dir', type=Path, default=Path('reports/tuning'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/final'))
    args = parser.parse_args()
    main(args.data, args.tuning_dir, args.output_dir)
