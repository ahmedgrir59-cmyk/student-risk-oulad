from __future__ import annotations
from typing import Any
import math


def _num(row: Any, key: str, default=float('nan')) -> float:
    try:
        v = row[key] if hasattr(row, '__getitem__') else getattr(row, key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def generate_recommendations(row: Any, cfg: dict, max_items: int = 4) -> list[dict]:
    """Support-oriented recommendations based only on actionable academic/engagement signals.

    Sensitive demographic attributes are deliberately not used to generate interventions.
    """
    recs: list[dict] = []

    missed = _num(row, 'assessments_missed_28d', 0)
    due = _num(row, 'assessments_due_28d', 0)
    completion = _num(row, 'assessment_completion_rate_28d')
    mean_score = _num(row, 'assessment_mean_score_28d')
    days_since = _num(row, 'days_since_last_activity_28d')
    active7 = _num(row, 'active_days_last7d', 0)
    clicks7 = _num(row, 'clicks_last7d', 0)
    slope = _num(row, 'weekly_click_slope_28d', 0)
    prev_attempts = _num(row, 'num_of_prev_attempts', 0)

    if missed > 0:
        recs.append({
            'factor': 'missed_assessments',
            'recommendation': 'Review missed or overdue assessments and agree on a catch-up plan.',
            'reason': f'{int(missed)} assessment(s) due in the observation window were not completed.'
        })
    elif due > 0 and not math.isnan(completion) and completion < cfg['completion_rate_low']:
        recs.append({
            'factor': 'low_assessment_completion',
            'recommendation': 'Prioritize the remaining early assessments and monitor completion closely.',
            'reason': f'Early assessment completion ({completion:.0%}) is below the development-data lower quartile.'
        })

    if not math.isnan(mean_score) and mean_score < cfg['assessment_score_low']:
        recs.append({
            'factor': 'low_early_assessment_score',
            'recommendation': 'Schedule targeted academic review on the topics covered by the early assessments.',
            'reason': f'Early mean assessment score ({mean_score:.1f}) is relatively low.'
        })

    if not math.isnan(days_since) and days_since >= cfg['days_since_activity_high']:
        recs.append({
            'factor': 'recent_inactivity',
            'recommendation': 'Use a supportive check-in to re-engage the student with current learning activities.',
            'reason': f'No recorded VLE activity for about {int(days_since)} day(s) before the prediction point.'
        })

    if active7 <= cfg['active_days_last7d_low'] or clicks7 <= cfg['clicks_last7d_low']:
        recs.append({
            'factor': 'low_recent_engagement',
            'recommendation': 'Encourage a short, regular weekly study schedule and monitor activity over the next week.',
            'reason': 'Recent engagement is in the lower range observed in the development data.'
        })

    if slope < cfg['weekly_click_slope_low']:
        recs.append({
            'factor': 'declining_engagement',
            'recommendation': 'Check for barriers causing declining engagement and agree on a recovery plan.',
            'reason': 'Weekly activity is trending downward across the first four weeks.'
        })

    if prev_attempts > 0:
        recs.append({
            'factor': 'previous_attempts',
            'recommendation': 'Consider additional tutoring or structured academic support based on prior attempt history.',
            'reason': f'The student has {int(prev_attempts)} previous attempt(s).'
        })

    # Remove duplicates by factor while preserving priority/order.
    seen = set(); out = []
    for r in recs:
        if r['factor'] not in seen:
            seen.add(r['factor']); out.append(r)
        if len(out) >= max_items:
            break
    if not out:
        out.append({
            'factor': 'monitor',
            'recommendation': 'Continue routine monitoring; no strong actionable warning signal was triggered by the rule set.',
            'reason': 'Current actionable indicators are not beyond the configured intervention thresholds.'
        })
    return out
