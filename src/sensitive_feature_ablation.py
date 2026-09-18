from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from split_utils import CV_SEED, DROP_FROM_X, GROUP, TARGET, indices_from_manifest, load_or_create_manifest
from tune_models import build_model, score_candidate

SENSITIVE = ['gender', 'age_band', 'disability', 'imd_band', 'highest_education', 'region']


def main(data: Path, tuning_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data)
    y = df[TARGET].astype(int)
    groups = df[GROUP]
    manifest = load_or_create_manifest(df, tuning_dir / 'locked_split_manifest.csv')
    dev_idx, holdout_idx = indices_from_manifest(manifest)
    selection = json.loads((tuning_dir / 'model_selection.json').read_text(encoding='utf-8'))
    params = selection['selected_params']

    rows = []
    for label, drop_extra in [('all_features', []), ('no_sensitive_demographics', SENSITIVE)]:
        drops = [c for c in DROP_FROM_X + drop_extra if c in df.columns]
        X = df.drop(columns=drops)
        X_dev = X.iloc[dev_idx].copy()
        y_dev = y.iloc[dev_idx].copy()
        g_dev = groups.iloc[dev_idx].copy()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
        pipe = build_model('logistic_regression', params, X_dev)
        cv_metrics = score_candidate(pipe, X_dev, y_dev, g_dev, cv)
        rows.append({
            'variant': label,
            'n_features_raw': int(X.shape[1]),
            **cv_metrics,
            'evaluation_scope': 'development_5fold_group_cv_only',
            'final_holdout_labels_accessed': False,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / 'sensitive_feature_ablation.csv', index=False)
    (output_dir / 'README.md').write_text(
        '# Sensitive-feature ablation\n\n'
        'Compares the selected Logistic Regression configuration with all permitted features versus a variant excluding gender, age_band, disability, imd_band, highest_education, and region. '
        'The diagnostic uses development-only 5-fold StratifiedGroupKFold CV and does not access locked-holdout labels or alter primary model selection.\n',
        encoding='utf-8',
    )
    print(out.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--tuning-dir', type=Path, default=Path('reports/tuning'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/ablation'))
    args = parser.parse_args()
    main(args.data, args.tuning_dir, args.output_dir)
