from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from split_utils import CV_SEED, DROP_FROM_X, GROUP, TARGET, indices_from_manifest, load_or_create_manifest

RANDOM_STATE = 42


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Leakage-safe preprocessing fitted inside each CV fold.

    Missingness is represented by domain features already present in the Day-28 table
    (assessment availability/submission counts, activity recency, etc.). We therefore do
    not auto-create per-column missing indicators, which avoids duplicated indicators for
    assessment score columns that share the same missingness mask.
    """
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), categorical),
    ])


def score_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    rows = []
    for train_idx, val_idx in cv.split(X, y, groups):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = fitted.predict(X.iloc[val_idx])
        if hasattr(fitted, "decision_function"):
            score = np.asarray(fitted.decision_function(X.iloc[val_idx]), dtype=float)
        else:
            score = np.asarray(fitted.predict_proba(X.iloc[val_idx])[:, 1], dtype=float)
        yy = y.iloc[val_idx]
        rows.append([
            accuracy_score(yy, pred),
            precision_score(yy, pred, zero_division=0),
            recall_score(yy, pred, zero_division=0),
            f1_score(yy, pred, zero_division=0),
            roc_auc_score(yy, score),
            average_precision_score(yy, score),
        ])

    arr = np.asarray(rows, dtype=float)
    metrics = {}
    for j, name in enumerate(["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
        metrics[f"{name}_mean"] = float(arr[:, j].mean())
        metrics[f"{name}_std"] = float(arr[:, j].std(ddof=1))
    return metrics


def main(data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Baselines are development-CV diagnostics only. No fitted baseline model artifacts
    # are persisted, and final-holdout labels are never accessed here.
    legacy_model_dir = output_dir / "models"
    if legacy_model_dir.exists():
        import shutil
        shutil.rmtree(legacy_model_dir)

    df = pd.read_csv(data_path)
    X = df.drop(columns=DROP_FROM_X)
    y = df[TARGET].astype(int)
    groups = df[GROUP]

    manifest = load_or_create_manifest(df, output_dir.parent / "tuning" / "locked_split_manifest.csv")
    dev_idx, holdout_idx = indices_from_manifest(manifest)
    X_dev = X.iloc[dev_idx].copy()
    y_dev = y.iloc[dev_idx].copy()
    groups_dev = groups.iloc[dev_idx].copy()
    groups_holdout = groups.iloc[holdout_idx].copy()

    assert set(groups_dev).isdisjoint(set(groups_holdout)), "Student leakage across development/final holdout"

    candidates = {
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(X_dev, scale_numeric=True)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Linear SVM": Pipeline([
            ("prep", make_preprocessor(X_dev, scale_numeric=True)),
            ("model", LinearSVC(C=1.0, class_weight="balanced", dual="auto", tol=1e-3, random_state=RANDOM_STATE, max_iter=5000)),
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor(X_dev, scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=80,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    rows = []
    for name, model in candidates.items():
        print(f"5-fold development CV baseline: {name}", flush=True)
        rows.append({"model": name, **score_cv(model, X_dev, y_dev, groups_dev)})

    result = pd.DataFrame(rows).sort_values(["f1_mean", "recall_mean", "roc_auc_mean"], ascending=False)
    result.to_csv(output_dir / "baseline_metrics.csv", index=False)

    split_summary = {
        "rows_total": int(len(df)),
        "unique_students_total": int(groups.nunique()),
        "development_rows": int(len(dev_idx)),
        "final_holdout_rows": int(len(holdout_idx)),
        "development_unique_students": int(groups_dev.nunique()),
        "final_holdout_unique_students": int(groups_holdout.nunique()),
        "student_overlap_development_holdout": 0,
        "development_at_risk_rate": float(y_dev.mean()),
        "model_features_before_encoding": int(X.shape[1]),
        "evaluation_scope": "development_5fold_group_cv_only",
        "final_holdout_labels_accessed": False,
        "note": "Baseline model comparison uses only development rows via 5-fold StratifiedGroupKFold. The final holdout is not scored here.",
    }
    (output_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baselines"))
    args = parser.parse_args()
    main(args.data, args.output_dir)
