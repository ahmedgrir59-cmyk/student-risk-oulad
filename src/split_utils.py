from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TARGET = 'target_at_risk'
GROUP = 'id_student'
DROP_FROM_X = [TARGET, GROUP, 'final_result']
HOLDOUT_SEED = 20260916
CV_SEED = 42


def locked_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Create the single canonical group-aware development/final-holdout split."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=HOLDOUT_SEED)
    dev_idx, test_idx = next(splitter.split(X, y, groups))
    assert set(groups.iloc[dev_idx]).isdisjoint(set(groups.iloc[test_idx]))
    return np.asarray(dev_idx), np.asarray(test_idx)


def split_manifest(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=DROP_FROM_X)
    y = df[TARGET].astype(int)
    g = df[GROUP]
    dev_idx, test_idx = locked_split(X, y, g)
    manifest = pd.DataFrame({
        'row_index': np.r_[dev_idx, test_idx],
        'split': ['development'] * len(dev_idx) + ['final_holdout'] * len(test_idx),
        'id_student': pd.concat([g.iloc[dev_idx], g.iloc[test_idx]], ignore_index=True),
    })
    return manifest


def load_or_create_manifest(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Use one persisted split everywhere. Refuse stale/incompatible manifests."""
    expected = split_manifest(df).sort_values('row_index').reset_index(drop=True)
    if path.exists():
        existing = pd.read_csv(path).sort_values('row_index').reset_index(drop=True)
        required = {'row_index', 'split', 'id_student'}
        if not required.issubset(existing.columns):
            raise ValueError(f'Locked split manifest missing columns: {required - set(existing.columns)}')
        if len(existing) != len(expected):
            raise ValueError('Locked split manifest length does not match current dataset.')
        if not existing[['row_index','split','id_student']].equals(expected[['row_index','split','id_student']]):
            raise ValueError('Locked split manifest does not match the canonical split for this dataset/seed.')
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    expected.to_csv(path, index=False)
    return expected


def indices_from_manifest(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dev = manifest.loc[manifest['split'].eq('development'), 'row_index'].to_numpy(dtype=int)
    test = manifest.loc[manifest['split'].eq('final_holdout'), 'row_index'].to_numpy(dtype=int)
    return dev, test
