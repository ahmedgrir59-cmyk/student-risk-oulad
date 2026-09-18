from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from split_utils import TARGET, GROUP, load_or_create_manifest, indices_from_manifest

KEY_COLUMNS = ['code_module', 'code_presentation', 'id_student']
AUDIT_ONLY = ['final_result']
WEEK_COLS = [f'clicks_week{i}_28d' for i in range(1, 5)]
SELECTED_CONTINUOUS = [
    'total_clicks_28d', 'active_days_28d', 'unique_sites_28d',
    'avg_clicks_per_active_day_28d', 'days_since_last_activity_28d',
    'clicks_last7d', 'active_days_last7d', 'weekly_click_slope_28d',
    'assessment_mean_score_28d', 'assessment_completion_rate_28d',
    'weighted_assessment_score_28d', 'assessments_missed_28d',
]
SELECTED_CATEGORICAL = [
    'gender', 'age_band', 'highest_education', 'imd_band', 'disability',
    'num_of_prev_attempts', 'code_module', 'code_presentation',
]


def save_current(fig_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_dir / name, dpi=160, bbox_inches='tight')
    plt.close()


def iqr_flag_table(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in data.columns:
            continue
        s = data[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((s < lo) | (s > hi)).sum())
        rows.append({
            'feature': col,
            'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr),
            'lower_bound': float(lo), 'upper_bound': float(hi),
            'iqr_flag_count': count,
            'iqr_flag_pct': float(count / len(s) * 100),
            'note': 'IQR flag only; not automatic evidence of bad data or reason to delete rows.',
        })
    return pd.DataFrame(rows).sort_values('iqr_flag_pct', ascending=False)


def main(data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    manifest_path = output_dir.parent / 'tuning' / 'locked_split_manifest.csv'
    manifest = load_or_create_manifest(df, manifest_path)
    train_idx, test_idx = indices_from_manifest(manifest)
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    # Full-dataset integrity audit: safe descriptive checks that do not tune the model.
    overview = {
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'unique_students': int(df[GROUP].nunique()),
        'duplicate_full_rows': int(df.duplicated().sum()),
        'duplicate_enrollment_keys': int(df.duplicated(KEY_COLUMNS).sum()),
        'target_at_risk_count': int(df[TARGET].sum()),
        'target_not_at_risk_count': int((df[TARGET] == 0).sum()),
        'target_at_risk_rate': float(df[TARGET].mean()),
        'train_rows_for_eda': int(len(train)),
        'test_rows_held_out': int(len(test)),
        'train_unique_students': int(train[GROUP].nunique()),
        'test_unique_students': int(test[GROUP].nunique()),
        'student_overlap_train_test': int(len(set(train[GROUP]).intersection(set(test[GROUP])))),
        'eda_policy': 'Integrity/missingness summaries may use the full master table; target-conditional EDA and feature/target relationships use only the persisted development partition. The locked evaluation holdout is excluded from target-facing EDA in the current reproducible pipeline.',
    }
    (output_dir / 'eda_overview.json').write_text(json.dumps(overview, indent=2), encoding='utf-8')

    # Missingness (full data: data-quality audit only).
    missing = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_pct': df.isna().mean() * 100,
        'dtype': df.dtypes.astype(str),
    }).sort_values(['missing_pct', 'missing_count'], ascending=False)
    missing.to_csv(output_dir / 'missingness.csv')

    nonzero_missing = missing[missing['missing_count'] > 0].head(20).sort_values('missing_pct')
    if not nonzero_missing.empty:
        plt.figure(figsize=(9, 5.5))
        plt.barh(nonzero_missing.index, nonzero_missing['missing_pct'])
        plt.xlabel('Missing values (%)')
        plt.ylabel('Feature')
        plt.title('Missingness in Day-28 Master Dataset')
        save_current(fig_dir, '01_missingness.png')

    # Class balance (train only for target-conditional reporting discipline).
    class_counts = train[TARGET].value_counts().sort_index()
    class_counts.index = ['Not At Risk (0)', 'At Risk (1)']
    class_counts.to_csv(output_dir / 'train_class_balance.csv', header=['count'])
    plt.figure(figsize=(6, 4.5))
    plt.bar(class_counts.index, class_counts.values)
    plt.ylabel('Training enrollments')
    plt.title('Training Class Balance')
    save_current(fig_dir, '02_train_class_balance.png')

    # Numeric descriptive statistics (train only for model-facing EDA).
    numeric_cols = [c for c in train.select_dtypes(include=np.number).columns if c not in [GROUP, TARGET]]
    numeric_summary = train[numeric_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    numeric_summary['missing_count'] = train[numeric_cols].isna().sum()
    numeric_summary['missing_pct'] = train[numeric_cols].isna().mean() * 100
    numeric_summary['skew'] = train[numeric_cols].skew(numeric_only=True)
    numeric_summary['zero_pct'] = (train[numeric_cols] == 0).mean() * 100
    numeric_summary.to_csv(output_dir / 'numeric_summary_train.csv')

    # Categorical summary (train only).
    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_rows = []
    for col in cat_cols:
        vc = train[col].value_counts(dropna=False)
        cat_rows.append({
            'feature': col,
            'unique_non_null': int(train[col].nunique(dropna=True)),
            'missing_count': int(train[col].isna().sum()),
            'missing_pct': float(train[col].isna().mean() * 100),
            'most_frequent_value': str(vc.index[0]) if len(vc) else '',
            'most_frequent_count': int(vc.iloc[0]) if len(vc) else 0,
        })
    pd.DataFrame(cat_rows).to_csv(output_dir / 'categorical_summary_train.csv', index=False)

    # IQR diagnostic flags. Do not automatically delete them.
    iqr_table = iqr_flag_table(train, [c for c in SELECTED_CONTINUOUS if c in train.columns])
    iqr_table.to_csv(output_dir / 'iqr_flags_train.csv', index=False)

    # Numeric association with target (training only). Pearson with binary y = point-biserial equivalent.
    assoc = train[numeric_cols + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
    assoc = assoc.sort_values(key=lambda s: s.abs(), ascending=False)
    assoc_df = assoc.rename('pearson_with_target').to_frame()
    assoc_df['abs_correlation'] = assoc_df['pearson_with_target'].abs()
    assoc_df.to_csv(output_dir / 'numeric_target_associations_train.csv')

    top_assoc = assoc_df.head(15).sort_values('pearson_with_target')
    plt.figure(figsize=(9, 6.5))
    plt.barh(top_assoc.index, top_assoc['pearson_with_target'])
    plt.axvline(0, linewidth=1)
    plt.xlabel('Pearson correlation with target_at_risk')
    plt.ylabel('Feature')
    plt.title('Top Numeric Associations with At-Risk Target (Training Only)')
    save_current(fig_dir, '03_top_numeric_target_associations.png')

    # Correlation matrix among strongest numeric signals, train only.
    corr_features = assoc_df.head(12).index.tolist()
    corr_matrix = train[corr_features + [TARGET]].corr(numeric_only=True)
    corr_matrix.to_csv(output_dir / 'selected_correlation_matrix_train.csv')
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix.values, aspect='auto', vmin=-1, vmax=1)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index, fontsize=8)
    plt.colorbar(im, label='Correlation')
    plt.title('Selected Correlation Matrix (Training Only)')
    save_current(fig_dir, '04_selected_correlation_matrix.png')

    # Weekly engagement trend by target (training only).
    weekly_means = train.groupby(TARGET)[WEEK_COLS].mean().T
    weekly_medians = train.groupby(TARGET)[WEEK_COLS].median().T
    weekly_means.index = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    weekly_medians.index = weekly_means.index
    weekly_means.columns = ['Not At Risk', 'At Risk']
    weekly_medians.columns = ['Not At Risk', 'At Risk']
    weekly_means.to_csv(output_dir / 'weekly_click_means_by_risk_train.csv')
    weekly_medians.to_csv(output_dir / 'weekly_click_medians_by_risk_train.csv')

    plt.figure(figsize=(7.5, 5))
    for c in weekly_means.columns:
        plt.plot(weekly_means.index, weekly_means[c], marker='o', label=c)
    plt.ylabel('Mean VLE clicks')
    plt.xlabel('Observation week')
    plt.title('Weekly Engagement Trend by Risk Class (Training Only)')
    plt.legend()
    save_current(fig_dir, '05_weekly_click_trend.png')

    # Distributions for selected important variables, train only.
    for idx, col in enumerate([
        'total_clicks_28d', 'active_days_28d', 'days_since_last_activity_28d',
        'assessment_mean_score_28d', 'assessment_completion_rate_28d'
    ], start=6):
        if col not in train.columns:
            continue
        plt.figure(figsize=(7.5, 5))
        for label, group in train.groupby(TARGET):
            vals = group[col].dropna().values
            if len(vals):
                # Use quantile clipping only for visualization so extreme tails do not hide the bulk.
                upper = np.nanquantile(train[col].dropna(), 0.99) if train[col].notna().any() else None
                if upper is not None and np.isfinite(upper):
                    vals = vals[vals <= upper]
                plt.hist(vals, bins=30, alpha=0.5, density=True, label='At Risk' if label == 1 else 'Not At Risk')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.title(f'{col} Distribution by Risk Class (Training Only)')
        plt.legend()
        save_current(fig_dir, f'{idx:02d}_{col}_distribution.png')

    # Risk rates by categorical features (training only, descriptive association, not causation).
    risk_tables = []
    fig_num = 11
    for col in SELECTED_CATEGORICAL:
        if col not in train.columns:
            continue
        table = train.groupby(col, dropna=False)[TARGET].agg(['mean', 'count']).reset_index()
        table = table.rename(columns={'mean': 'risk_rate', 'count': 'n'})
        table['feature'] = col
        risk_tables.append(table[['feature', col, 'risk_rate', 'n']].rename(columns={col: 'category'}))
        # Avoid unreadable charts with too many categories.
        if len(table) <= 12:
            plot_table = table.sort_values('risk_rate')
            plt.figure(figsize=(8, max(4.5, 0.38 * len(plot_table) + 2)))
            plt.barh(plot_table[col].astype(str), plot_table['risk_rate'] * 100)
            plt.xlabel('At-risk rate (%)')
            plt.ylabel(col)
            plt.title(f'At-Risk Rate by {col} (Training Only)')
            save_current(fig_dir, f'{fig_num:02d}_risk_rate_by_{col}.png')
            fig_num += 1
    if risk_tables:
        pd.concat(risk_tables, ignore_index=True).to_csv(output_dir / 'categorical_risk_rates_train.csv', index=False)

    # Group means/medians for strongest numeric features.
    compare_cols = [c for c in assoc_df.head(20).index if c in train.columns]
    means = train.groupby(TARGET)[compare_cols].mean().T
    medians = train.groupby(TARGET)[compare_cols].median().T
    means.columns = ['not_at_risk_mean', 'at_risk_mean']
    medians.columns = ['not_at_risk_median', 'at_risk_median']
    comparison = means.join(medians)
    comparison['mean_difference_at_risk_minus_not'] = comparison['at_risk_mean'] - comparison['not_at_risk_mean']
    comparison.to_csv(output_dir / 'top_numeric_group_comparison_train.csv')

    # Key findings generated from actual training data.
    assoc_top = assoc_df.head(8)
    highest_missing = missing[missing['missing_count'] > 0].head(8)
    risk_by_edu = train.groupby('highest_education')[TARGET].agg(['mean', 'count']).sort_values('mean', ascending=False)
    risk_by_attempt = train.groupby('num_of_prev_attempts')[TARGET].agg(['mean', 'count']).sort_index()

    findings = []
    findings.append('# Full EDA — Day-28 Student Risk Dataset\n')
    findings.append('## Scope and leakage discipline\n')
    findings.append(
        'Dataset integrity and missingness are summarized on the complete master table. '
        'All target-conditional comparisons, correlations, and risk-rate analyses below use **training rows only**, '
        'using the single canonical locked student-group-aware development/final-holdout split shared by EDA, tuning, calibration, and final evaluation. The locked evaluation holdout rows are not used to choose features, thresholds, or models.\n'
    )
    findings.append('## Dataset integrity\n')
    findings.append(f"- Master rows: **{overview['rows']:,}**; columns: **{overview['columns']}**; unique students: **{overview['unique_students']:,}**.")
    findings.append(f"- Duplicate full rows: **{overview['duplicate_full_rows']}**; duplicate enrollment keys: **{overview['duplicate_enrollment_keys']}**.")
    findings.append(f"- Development rows used for target-facing EDA: **{overview['train_rows_for_eda']:,}**; locked holdout rows excluded from target-facing EDA: **{overview['test_rows_held_out']:,}**.")
    findings.append(f"- Student overlap between development and holdout: **{overview['student_overlap_train_test']}**.\n")
    findings.append('## Missingness\n')
    if highest_missing.empty:
        findings.append('- No missing values were found.\n')
    else:
        for feature, row in highest_missing.iterrows():
            findings.append(f"- `{feature}`: {int(row['missing_count']):,} missing ({row['missing_pct']:.1f}%).")
        findings.append('\nMissing assessment-derived values are expected for students who do not yet have a usable early assessment/score by Day 28; they should be imputed inside the training pipeline rather than filled using the full dataset.\n')

    findings.append('## Strongest numeric associations with the at-risk label (development only)\n')
    findings.append('These are descriptive associations, **not causal effects**. Negative values mean higher feature values are associated with lower observed risk in this dataset.\n')
    for feature, row in assoc_top.iterrows():
        findings.append(f"- `{feature}`: correlation **{row['pearson_with_target']:+.3f}**.")

    findings.append('\n## Weekly engagement pattern (development only)\n')
    for week in weekly_means.index:
        findings.append(
            f"- {week}: mean clicks = **{weekly_means.loc[week, 'Not At Risk']:.1f}** for not-at-risk vs "
            f"**{weekly_means.loc[week, 'At Risk']:.1f}** for at-risk enrollments."
        )

    findings.append('\n## Selected categorical patterns (development only)\n')
    for cat, row in risk_by_edu.iterrows():
        findings.append(f"- Education `{cat}`: observed risk rate **{row['mean']*100:.1f}%** (n={int(row['count']):,}).")
    findings.append('\nPrevious attempts also show a strong descriptive gradient, but the highest attempt counts contain very small samples and must not be overinterpreted.\n')
    for attempts, row in risk_by_attempt.iterrows():
        findings.append(f"- Previous attempts `{attempts}`: risk **{row['mean']*100:.1f}%** (n={int(row['count']):,}).")

    findings.append('\n## Outliers\n')
    findings.append(
        'IQR flags were calculated only as diagnostics. Engagement/click variables are naturally right-skewed, '
        'so a high IQR-flag count is **not** a reason to delete students automatically. Any outlier treatment must be justified by impossible/invalid values or validated downstream performance, not by the IQR rule alone.\n'
    )

    findings.append('## EDA conclusion\n')
    findings.append(
        'The Day-28 dataset contains useful early signal in assessment completion/performance and engagement behavior. '
        'The next stage should keep the current leakage-safe preprocessing pipelines and perform student-group-aware 5-fold cross-validation and hyperparameter tuning on the training partition only. '
        'No locked evaluation holdout metric should be used for tuning or threshold selection.'
    )
    (output_dir / 'EDA_REPORT.md').write_text('\n'.join(findings), encoding='utf-8')

    print(json.dumps(overview, indent=2))
    print('\nTop numeric associations (training only):')
    print(assoc_df.head(12).to_string())
    print(f'\nEDA artifacts written to: {output_dir}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Leakage-disciplined EDA for the Day-28 OULAD risk dataset.')
    p.add_argument('data', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('reports/eda'))
    args = p.parse_args()
    main(args.data, args.output_dir)
