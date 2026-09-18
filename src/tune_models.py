from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from split_utils import CV_SEED, DROP_FROM_X, GROUP, TARGET, indices_from_manifest, load_or_create_manifest


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    nsteps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        nsteps.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(nsteps), num),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
        ]), cat),
    ])


def score_candidate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv: StratifiedGroupKFold) -> dict:
    """Generic development-CV scorer used by diagnostic scripts."""
    vals = []
    for tr, va in cv.split(X, y, groups):
        m = clone(pipe)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[va])
        score = m.decision_function(X.iloc[va]) if hasattr(m, 'decision_function') else m.predict_proba(X.iloc[va])[:, 1]
        yy = y.iloc[va]
        vals.append([
            accuracy_score(yy, pred), precision_score(yy, pred, zero_division=0),
            recall_score(yy, pred, zero_division=0), f1_score(yy, pred, zero_division=0),
            roc_auc_score(yy, score), average_precision_score(yy, score),
        ])
    return summarize_metrics(np.asarray(vals, dtype=float))


def summarize_metrics(arr: np.ndarray) -> dict:
    out = {}
    for j, name in enumerate(['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']):
        out[f'{name}_mean'] = float(arr[:, j].mean())
        out[f'{name}_std'] = float(arr[:, j].std(ddof=1))
    return out


def build_model(name: str, params: dict, X: pd.DataFrame) -> Pipeline:
    if name == 'logistic_regression':
        estimator = LogisticRegression(**params, solver='liblinear', max_iter=1500, random_state=CV_SEED)
        return Pipeline([('prep', make_preprocessor(X, True)), ('model', estimator)])
    if name == 'linear_svm':
        estimator = LinearSVC(**params, dual='auto', tol=1e-3, max_iter=7000, random_state=CV_SEED)
        return Pipeline([('prep', make_preprocessor(X, True)), ('model', estimator)])
    if name == 'random_forest':
        estimator = RandomForestClassifier(**params, n_jobs=-1, random_state=CV_SEED)
        return Pipeline([('prep', make_preprocessor(X, False)), ('model', estimator)])
    raise ValueError(name)


def estimator_for(name: str, params: dict):
    if name == 'logistic_regression':
        return LogisticRegression(**params, solver='liblinear', max_iter=1500, random_state=CV_SEED)
    if name == 'linear_svm':
        return LinearSVC(**params, dual='auto', tol=1e-3, max_iter=7000, random_state=CV_SEED)
    return RandomForestClassifier(**params, n_jobs=-1, random_state=CV_SEED)


def score_on_cached_folds(name: str, params: dict, fold_cache: list[dict]) -> dict:
    vals = []
    use_scaled = name in {'logistic_regression', 'linear_svm'}
    for fold in fold_cache:
        Xtr = fold['scaled_train'] if use_scaled else fold['unscaled_train']
        Xva = fold['scaled_val'] if use_scaled else fold['unscaled_val']
        ytr, yva = fold['y_train'], fold['y_val']
        est = estimator_for(name, params)
        est.fit(Xtr, ytr)
        pred = est.predict(Xva)
        score = est.decision_function(Xva) if hasattr(est, 'decision_function') else est.predict_proba(Xva)[:, 1]
        vals.append([
            accuracy_score(yva, pred), precision_score(yva, pred, zero_division=0),
            recall_score(yva, pred, zero_division=0), f1_score(yva, pred, zero_division=0),
            roc_auc_score(yva, score), average_precision_score(yva, score),
        ])
    return summarize_metrics(np.asarray(vals, dtype=float))


def main(data: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    candidate_dir = outdir / 'candidate_models'
    candidate_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data)
    X = df.drop(columns=DROP_FROM_X)
    y = df[TARGET].astype(int)
    groups = df[GROUP]
    manifest = load_or_create_manifest(df, outdir / 'locked_split_manifest.csv')
    dev_idx, holdout_idx = indices_from_manifest(manifest)
    X_dev, y_dev, g_dev = X.iloc[dev_idx].copy(), y.iloc[dev_idx].copy(), groups.iloc[dev_idx].copy()
    g_holdout = groups.iloc[holdout_idx]
    assert set(g_dev).isdisjoint(set(g_holdout))

    candidates = {
        'logistic_regression': [
            {'C': 0.05, 'class_weight': 'balanced'},
            {'C': 0.1, 'class_weight': 'balanced'},
            {'C': 0.3, 'class_weight': 'balanced'},
            {'C': 1.0, 'class_weight': 'balanced'},
            {'C': 0.1, 'class_weight': None},
            {'C': 1.0, 'class_weight': None},
        ],
        'linear_svm': [{'C': c, 'class_weight': 'balanced'} for c in [0.01, 0.03, 0.1, 0.3]],
        'random_forest': [
            {'n_estimators': 80, 'max_depth': 12, 'min_samples_leaf': 4, 'min_samples_split': 8, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'},
            {'n_estimators': 80, 'max_depth': 20, 'min_samples_leaf': 3, 'min_samples_split': 6, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample'},
        ],
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    splits = list(cv.split(X_dev, y_dev, g_dev))
    print('Preparing fold-specific preprocessing caches...', flush=True)
    fold_cache = []
    for k, (tr, va) in enumerate(splits, 1):
        Xtr, Xva = X_dev.iloc[tr], X_dev.iloc[va]
        scaled = make_preprocessor(Xtr, True)
        unscaled = make_preprocessor(Xtr, False)
        fold_cache.append({
            'scaled_train': scaled.fit_transform(Xtr),
            'scaled_val': scaled.transform(Xva),
            'unscaled_train': unscaled.fit_transform(Xtr),
            'unscaled_val': unscaled.transform(Xva),
            'y_train': y_dev.iloc[tr].to_numpy(),
            'y_val': y_dev.iloc[va].to_numpy(),
        })
        print(f'  prepared fold {k}/5', flush=True)

    best_rows = []
    for name, param_list in candidates.items():
        result_path = outdir / f'{name}_tuning.csv'
        model_path = candidate_dir / f'{name}_tuned_dev.joblib'
        if result_path.exists() and model_path.exists():
            print(f'Reusing completed tuning results for {name}', flush=True)
            result = pd.read_csv(result_path).sort_values(['f1_mean', 'recall_mean', 'roc_auc_mean'], ascending=False)
        else:
            rows = []
            for params in param_list:
                print(name, params, flush=True)
                metrics = score_on_cached_folds(name, params, fold_cache)
                rows.append({'model': name, 'params_json': json.dumps(params, sort_keys=True), **metrics})
            result = pd.DataFrame(rows).sort_values(['f1_mean', 'recall_mean', 'roc_auc_mean'], ascending=False)
            result.to_csv(result_path, index=False)
            best_params = json.loads(result.iloc[0].params_json)
            fitted = build_model(name, best_params, X_dev)
            fitted.fit(X_dev, y_dev)
            joblib.dump(fitted, model_path)
        best_rows.append(result.iloc[0])

    summary = pd.DataFrame(best_rows).sort_values(['f1_mean', 'recall_mean', 'roc_auc_mean'], ascending=False).reset_index(drop=True)
    summary['selection_rank'] = np.arange(1, len(summary) + 1)
    summary.to_csv(outdir / 'tuned_model_cv_summary.csv', index=False)

    selected = summary.iloc[0]
    selection = {
        'selection_rule': 'Highest mean 5-fold StratifiedGroupKFold F1 on development data; recall then ROC-AUC break ties.',
        'selected_model': selected.model,
        'selected_params': json.loads(selected.params_json),
        'selected_cv_f1': float(selected.f1_mean),
        'selected_cv_recall': float(selected.recall_mean),
        'selected_cv_roc_auc': float(selected.roc_auc_mean),
        'selected_cv_pr_auc': float(selected.pr_auc_mean),
        'development_rows': int(len(dev_idx)),
        'final_holdout_rows': int(len(holdout_idx)),
        'development_unique_students': int(g_dev.nunique()),
        'final_holdout_unique_students': int(g_holdout.nunique()),
        'student_overlap': 0,
        'final_holdout_target_not_used_during_tuning': True,
        'holdout_methodology_note': 'The current reproducible tuning/calibration/threshold-selection pipeline does not use final-holdout targets. During earlier iterative project development, holdout diagnostics were viewed, so the packaged holdout is a locked evaluation holdout rather than a pristine never-seen external test. External/future-cohort validation remains recommended.',
    }
    (outdir / 'model_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/tuning'))
    args = parser.parse_args()
    main(args.data, args.output_dir)
