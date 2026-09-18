from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_DAY = 28
OBS_START = 0
OBS_END = 28
KEYS = ["code_module", "code_presentation", "id_student"]
RISK_MAP = {"Fail": 1, "Withdrawn": 1, "Pass": 0, "Distinction": 0}


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return value.strip("_")


def read_csv_from_zip(z: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with z.open(name) as f:
        return pd.read_csv(f, **kwargs)


def build_dataset(zip_path: Path, output_csv: Path, report_json: Path, dictionary_csv: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        info = read_csv_from_zip(z, "studentInfo.csv")
        reg = read_csv_from_zip(z, "studentRegistration.csv")
        courses = read_csv_from_zip(z, "courses.csv")
        assessments = read_csv_from_zip(z, "assessments.csv")
        student_assess = read_csv_from_zip(z, "studentAssessment.csv")
        vle_meta = read_csv_from_zip(z, "vle.csv")

        # ------------------------------------------------------------------
        # 1) Cohort definition at prediction day
        # ------------------------------------------------------------------
        base = info.merge(reg, on=KEYS, how="left", validate="one_to_one")
        base = base.merge(
            courses,
            on=["code_module", "code_presentation"],
            how="left",
            validate="many_to_one",
        )

        initial_n = len(base)
        missing_registration = int(base["date_registration"].isna().sum())
        registered_after_cutoff = int((base["date_registration"] > PREDICTION_DAY).fillna(False).sum())
        withdrawn_by_cutoff = int((base["date_unregistration"] <= PREDICTION_DAY).fillna(False).sum())

        eligible = base[
            base["date_registration"].notna()
            & (base["date_registration"] <= PREDICTION_DAY)
            & (base["date_unregistration"].isna() | (base["date_unregistration"] > PREDICTION_DAY))
        ].copy()

        eligible["target_at_risk"] = eligible["final_result"].map(RISK_MAP)
        if eligible["target_at_risk"].isna().any():
            bad = eligible.loc[eligible["target_at_risk"].isna(), "final_result"].unique().tolist()
            raise ValueError(f"Unexpected final_result values: {bad}")
        eligible["target_at_risk"] = eligible["target_at_risk"].astype("int8")

        # Keep ID only as a grouping / audit key. It must NOT be a model feature.
        static_cols = KEYS + [
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "date_registration",
            "module_presentation_length",
            "target_at_risk",
            "final_result",  # audit only; explicitly excluded from model features
        ]
        dataset = eligible[static_cols].copy()
        eligible_keys = eligible[KEYS].drop_duplicates()

        # ------------------------------------------------------------------
        # 2) VLE behavior features, only from days 0..28
        # ------------------------------------------------------------------
        # Filter in chunks so the 10M+ row file is handled safely.
        vle_parts = []
        with z.open("studentVle.csv") as f:
            for chunk in pd.read_csv(f, chunksize=750_000):
                chunk = chunk[(chunk["date"] >= OBS_START) & (chunk["date"] <= OBS_END)]
                if chunk.empty:
                    continue
                chunk = chunk.merge(eligible_keys, on=KEYS, how="inner")
                if not chunk.empty:
                    vle_parts.append(chunk)

        if vle_parts:
            sv = pd.concat(vle_parts, ignore_index=True)
            sv = sv.merge(
                vle_meta[["id_site", "code_module", "code_presentation", "activity_type"]],
                on=["id_site", "code_module", "code_presentation"],
                how="left",
                validate="many_to_one",
            )

            g = sv.groupby(KEYS, sort=False)
            vle_basic = g.agg(
                total_clicks_28d=("sum_click", "sum"),
                active_days_28d=("date", "nunique"),
                unique_sites_28d=("id_site", "nunique"),
                unique_activity_types_28d=("activity_type", "nunique"),
                last_activity_day_28d=("date", "max"),
            ).reset_index()
            vle_basic["avg_clicks_per_active_day_28d"] = (
                vle_basic["total_clicks_28d"] / vle_basic["active_days_28d"].replace(0, np.nan)
            )
            vle_basic["days_since_last_activity_28d"] = PREDICTION_DAY - vle_basic["last_activity_day_28d"]

            # Weekly click totals. Week 4 includes day 28 so all information up to the prediction point is retained.
            bins = [OBS_START - 1, 6, 13, 20, OBS_END]
            labels = ["week1", "week2", "week3", "week4"]
            sv["obs_week"] = pd.cut(sv["date"], bins=bins, labels=labels)
            weekly = (
                sv.pivot_table(
                    index=KEYS,
                    columns="obs_week",
                    values="sum_click",
                    aggfunc="sum",
                    fill_value=0,
                    observed=False,
                )
                .rename(columns=lambda x: f"clicks_{x}_28d")
                .reset_index()
            )
            for w in labels:
                col = f"clicks_{w}_28d"
                if col not in weekly.columns:
                    weekly[col] = 0
            x = np.arange(1, 5, dtype=float)
            x_center = x - x.mean()
            denom = np.sum(x_center**2)
            y = weekly[[f"clicks_{w}_28d" for w in labels]].to_numpy(dtype=float)
            weekly["weekly_click_slope_28d"] = (y @ x_center) / denom
            weekly["click_change_week4_vs_week1"] = weekly["clicks_week4_28d"] - weekly["clicks_week1_28d"]

            # Recent behavior in the last seven days (days 22..28 inclusive).
            recent = sv[sv["date"] >= 22].groupby(KEYS, sort=False).agg(
                clicks_last7d=("sum_click", "sum"),
                active_days_last7d=("date", "nunique"),
            ).reset_index()

            # Activity-type totals are highly interpretable for SHAP and recommendations.
            activity = (
                sv.pivot_table(
                    index=KEYS,
                    columns="activity_type",
                    values="sum_click",
                    aggfunc="sum",
                    fill_value=0,
                )
                .rename(columns=lambda c: f"vle_clicks_{safe_name(c)}_28d")
                .reset_index()
            )

            for frame in [vle_basic, weekly, recent, activity]:
                dataset = dataset.merge(frame, on=KEYS, how="left", validate="one_to_one")

        # Fill absence of VLE records with zero only for behavioral count-like features.
        vle_zero_prefixes = ("total_clicks_", "active_days_", "unique_sites_", "unique_activity_types_", "clicks_", "vle_clicks_")
        vle_zero_exact = {"weekly_click_slope_28d", "click_change_week4_vs_week1", "avg_clicks_per_active_day_28d"}
        for col in dataset.columns:
            if col.startswith(vle_zero_prefixes) or col in vle_zero_exact:
                dataset[col] = dataset[col].fillna(0)
        if "last_activity_day_28d" in dataset:
            # No activity is represented by NaN last-day and 29 days since activity.
            dataset["days_since_last_activity_28d"] = dataset["days_since_last_activity_28d"].fillna(PREDICTION_DAY + 1)

        # ------------------------------------------------------------------
        # 3) Early assessment features available by prediction day
        # ------------------------------------------------------------------
        due_early = assessments[assessments["date"].notna() & (assessments["date"] <= PREDICTION_DAY)].copy()
        due_counts = due_early.groupby(["code_module", "code_presentation"]).agg(
            assessments_due_28d=("id_assessment", "nunique"),
            assessment_weight_due_28d=("weight", "sum"),
        ).reset_index()

        sa = student_assess.merge(
            assessments[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]],
            on="id_assessment",
            how="left",
            validate="many_to_one",
        )
        # Strict leakage rule for score-based features:
        # only assessments whose official due date is <= day 28 AND submitted <= day 28.
        sa = sa[
            sa["date"].notna()
            & (sa["date"] <= PREDICTION_DAY)
            & (sa["date_submitted"] <= PREDICTION_DAY)
        ].copy()
        sa = sa.merge(eligible_keys, on=KEYS, how="inner")

        if not sa.empty:
            sa["is_late"] = (sa["date_submitted"] > sa["date"]).astype(int)
            sa["weighted_score_component"] = sa["score"] * sa["weight"]
            ag = sa.groupby(KEYS, sort=False).agg(
                assessments_submitted_28d=("id_assessment", "nunique"),
                assessment_mean_score_28d=("score", "mean"),
                assessment_min_score_28d=("score", "min"),
                assessment_max_score_28d=("score", "max"),
                late_submissions_28d=("is_late", "sum"),
                banked_submissions_28d=("is_banked", "sum"),
                assessment_weight_completed_28d=("weight", "sum"),
                weighted_score_sum_28d=("weighted_score_component", "sum"),
            ).reset_index()
            ag["weighted_assessment_score_28d"] = (
                ag["weighted_score_sum_28d"] / ag["assessment_weight_completed_28d"].replace(0, np.nan)
            )
            ag = ag.drop(columns=["weighted_score_sum_28d"])
            dataset = dataset.merge(ag, on=KEYS, how="left", validate="one_to_one")

        dataset = dataset.merge(
            due_counts,
            on=["code_module", "code_presentation"],
            how="left",
            validate="many_to_one",
        )
        for col in [
            "assessments_due_28d",
            "assessment_weight_due_28d",
            "assessments_submitted_28d",
            "late_submissions_28d",
            "banked_submissions_28d",
            "assessment_weight_completed_28d",
        ]:
            if col in dataset:
                dataset[col] = dataset[col].fillna(0)

        dataset["assessments_missed_28d"] = (
            dataset["assessments_due_28d"] - dataset["assessments_submitted_28d"]
        ).clip(lower=0)
        dataset["assessment_completion_rate_28d"] = np.where(
            dataset["assessments_due_28d"] > 0,
            dataset["assessments_submitted_28d"] / dataset["assessments_due_28d"],
            np.nan,
        )
        dataset["has_early_assessment_28d"] = (dataset["assessments_due_28d"] > 0).astype("int8")

        # ------------------------------------------------------------------
        # 4) Assertions and outputs
        # ------------------------------------------------------------------
        if dataset.duplicated(KEYS).any():
            raise AssertionError("Duplicate enrollment keys found in final dataset.")

        # Model must never receive these columns directly.
        leakage_or_group_only = {
            "id_student": "group/audit identifier; exclude from X",
            "final_result": "source of target; direct leakage; exclude from X",
        }

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        report_json.parent.mkdir(parents=True, exist_ok=True)
        dictionary_csv.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_csv, index=False)

        target_counts = dataset["target_at_risk"].value_counts().sort_index().to_dict()
        report = {
            "prediction_day": PREDICTION_DAY,
            "observation_window": [OBS_START, OBS_END],
            "target_definition": {"1": ["Fail", "Withdrawn"], "0": ["Pass", "Distinction"]},
            "source_enrollments": initial_n,
            "excluded_missing_registration_date": missing_registration,
            "excluded_registered_after_day28": registered_after_cutoff,
            "excluded_unregistered_by_day28": withdrawn_by_cutoff,
            "final_eligible_rows": int(len(dataset)),
            "unique_students": int(dataset["id_student"].nunique()),
            "target_counts": {str(k): int(v) for k, v in target_counts.items()},
            "target_at_risk_rate": float(dataset["target_at_risk"].mean()),
            "duplicate_enrollment_keys": int(dataset.duplicated(KEYS).sum()),
            "columns": int(dataset.shape[1]),
            "leakage_or_group_only_columns": leakage_or_group_only,
            "notes": [
                "Only students still registered at prediction day are included.",
                "VLE behavior uses days 0 through 28 only.",
                "Assessment score features use assessments due by day 28 and submitted by day 28.",
                "id_student is retained only for group-aware splitting and must not enter the model.",
                "final_result is retained only for audit and must not enter the model.",
            ],
        }
        report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        rows = []
        for col in dataset.columns:
            role = "feature"
            note = ""
            if col == "target_at_risk":
                role = "target"
                note = "1=Fail/Withdrawn, 0=Pass/Distinction"
            elif col == "id_student":
                role = "group_id"
                note = leakage_or_group_only[col]
            elif col == "final_result":
                role = "audit_only"
                note = leakage_or_group_only[col]
            elif col in ["code_module", "code_presentation"]:
                role = "categorical_feature_and_key"
            rows.append({
                "column": col,
                "dtype": str(dataset[col].dtype),
                "role": role,
                "missing_count": int(dataset[col].isna().sum()),
                "note": note,
            })
        pd.DataFrame(rows).to_csv(dictionary_csv, index=False)
        return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build leakage-aware OULAD day-28 student risk dataset.")
    parser.add_argument("zip_path", type=Path, help="Path to OULAD zip archive")
    parser.add_argument("--output", type=Path, default=Path("data/processed/student_risk_day28.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/day28_dataset_summary.json"))
    parser.add_argument("--dictionary", type=Path, default=Path("reports/feature_dictionary.csv"))
    args = parser.parse_args()

    df = build_dataset(args.zip_path, args.output, args.report, args.dictionary)
    print(f"Built dataset: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"At-risk rate: {df['target_at_risk'].mean():.2%}")
