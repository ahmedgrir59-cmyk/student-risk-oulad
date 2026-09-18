from __future__ import annotations

import json
import math
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORTS = PROJECT_ROOT / "reports" / "final"
MASTER_CSV = PROJECT_ROOT / "data" / "processed" / "student_risk_day28.csv"
PROGRESS_CSV = REPORTS / "progress_monitoring_view.csv"
METRICS_JSON = REPORTS / "final_test_metrics.json"
THRESHOLDS_JSON = REPORTS / "risk_thresholds.json"
BUNDLE_PATH = REPORTS / "final_risk_bundle.joblib"
GLOBAL_SHAP_CSV = REPORTS / "shap_global_importance.csv"
FAIRNESS_CSV = REPORTS / "fairness_support_audit.csv"
MODEL_CV_CSV = PROJECT_ROOT / "reports" / "tuning" / "tuned_model_cv_summary.csv"
SPLIT_MANIFEST_CSV = PROJECT_ROOT / "reports" / "tuning" / "locked_split_manifest.csv"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from recommendation_engine import generate_recommendations  # noqa: E402


def _raw_score(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _risk_level(p: float, low: float, high: float) -> str:
    return "High" if p >= high else ("Medium" if p >= low else "Low")



def _source_feature_name(transformed_feature: str, raw_columns: list[str]) -> str:
    if transformed_feature.startswith("num__"):
        return transformed_feature[len("num__"): ]
    if transformed_feature.startswith("cat__"):
        tail = transformed_feature[len("cat__"): ]
        for col in sorted(raw_columns, key=len, reverse=True):
            if tail == col or tail.startswith(col + "_"):
                return col
    return transformed_feature


def _original_value_for_feature(transformed_feature: str, row: pd.Series, raw_columns: list[str]):
    src = _source_feature_name(transformed_feature, raw_columns)
    if src in row.index:
        val = row[src]
        if pd.isna(val):
            return None
        return _safe(val)
    return None


def _safe(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


class DataStore:
    def __init__(self) -> None:
        required = [
            PROGRESS_CSV, METRICS_JSON, THRESHOLDS_JSON, BUNDLE_PATH, MASTER_CSV,
            GLOBAL_SHAP_CSV, FAIRNESS_CSV, MODEL_CV_CSV, SPLIT_MANIFEST_CSV,
        ]
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Demo cannot start; missing required artifact(s): " + ", ".join(missing))

        self.progress = pd.read_csv(PROGRESS_CSV)
        self.metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        self.thresholds = json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8"))
        self.bundle = joblib.load(BUNDLE_PATH)
        self.master = pd.read_csv(MASTER_CSV)
        self.master.index.name = "row_index"
        self.master["row_index"] = self.master.index.astype(int)
        self.global_shap = pd.read_csv(GLOBAL_SHAP_CSV).head(12)
        self.fairness = pd.read_csv(FAIRNESS_CSV)
        self.model_cv = pd.read_csv(MODEL_CV_CSV)
        self.split_manifest = pd.read_csv(SPLIT_MANIFEST_CSV)

        # Components needed for on-demand local explanations. The explainer is built
        # lazily from the same deterministic 500-row development background used by
        # finalize_model.py. This does not refit the model or touch holdout labels.
        self.feature_columns = list(self.bundle["feature_columns"])
        self.prep = self.bundle["base_model"].named_steps["prep"]
        self.estimator = self.bundle["base_model"].named_steps["model"]
        self.transformed_feature_names = np.asarray(self.prep.get_feature_names_out(), dtype=str)
        self._local_explainer = None
        self._local_cache = {}

        # Fast lookups.
        self.progress["student_key"] = (
            self.progress["id_student"].astype(str)
            + "|"
            + self.progress["code_module"].astype(str)
            + "|"
            + self.progress["code_presentation"].astype(str)
        )
        self.master_by_row = self.master.set_index("row_index", drop=False)

    def health(self):
        return {
            "status": "ok",
            "project": "student-risk-oulad",
            "predictionPointDay": int(self.bundle["prediction_point_day"]),
            "model": self.metrics["selected_model"],
            "rowsLoaded": int(len(self.progress)),
            "studentsLoaded": int(self.progress["id_student"].nunique()),
        }

    def summary(self):
        p = self.progress
        risk_counts = p["risk_level"].value_counts().to_dict()
        risk_rates = p.groupby("risk_level")["actual_at_risk"].mean().to_dict()
        tuned = self.metrics["tuned_threshold_metrics"]
        return {
            "totalStudents": int(p["id_student"].nunique()),
            "totalRows": int(len(p)),
            "riskCounts": {k: int(v) for k, v in risk_counts.items()},
            "riskActualRates": {k: float(v) for k, v in risk_rates.items()},
            "model": {
                "name": self.metrics["selected_model"].replace("_", " ").title(),
                "recall": float(tuned["recall"]),
                "f1": float(tuned["f1"]),
                "rocAuc": float(tuned["roc_auc"]),
                "prAuc": float(tuned["pr_auc"]),
                "accuracy": float(tuned["accuracy"]),
                "threshold": float(tuned["threshold"]),
            },
            "thresholds": self.thresholds,
        }

    def students(self, risk="All", search="", limit=80, sort="prob_desc"):
        d = self.progress
        if risk in {"Low", "Medium", "High"}:
            d = d[d["risk_level"] == risk]
        if search:
            s = search.lower().strip()
            mask = (
                d["id_student"].astype(str).str.contains(s, regex=False)
                | d["code_module"].astype(str).str.lower().str.contains(s, regex=False)
                | d["code_presentation"].astype(str).str.lower().str.contains(s, regex=False)
            )
            d = d[mask]
        ascending = sort == "prob_asc"
        d = d.sort_values("risk_probability", ascending=ascending).head(max(1, min(int(limit), 250)))
        cols = [
            "row_index", "id_student", "code_module", "code_presentation",
            "risk_probability", "risk_level", "predicted_at_risk",
            "actual_at_risk", "engagement_direction",
        ]
        return [{k: _safe(v) for k, v in r.items()} for r in d[cols].to_dict("records")]

    def _ensure_local_explainer(self):
        if self._local_explainer is not None:
            return
        dev_rows = self.split_manifest.loc[
            self.split_manifest["split"] == "development", "row_index"
        ].astype(int).to_numpy()
        X_dev = self.master_by_row.loc[dev_rows, self.feature_columns]
        rng = np.random.default_rng(42)
        bg_pos = rng.choice(len(X_dev), size=min(500, len(X_dev)), replace=False)
        X_bg = self.prep.transform(X_dev.iloc[bg_pos])
        self._local_explainer = shap.LinearExplainer(self.estimator, X_bg)

    def _local_explain(self, row_index: int):
        if row_index in self._local_cache:
            return self._local_cache[row_index]
        if row_index not in self.master_by_row.index:
            return None

        self._ensure_local_explainer()
        row = self.master_by_row.loc[row_index]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        X = pd.DataFrame([row[self.feature_columns].to_dict()])
        X_t = self.prep.transform(X)
        X_dense = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
        sv = self._local_explainer(X_t)
        vals = np.asarray(sv.values[0], dtype=float)

        pos = np.where(vals > 0)[0]
        pos = pos[np.argsort(vals[pos])[::-1]][:5] if len(pos) else []
        neg = np.where(vals < 0)[0]
        neg = neg[np.argsort(vals[neg])][:5] if len(neg) else []

        def pack(indices):
            items = []
            for j in indices:
                transformed = str(self.transformed_feature_names[j])
                items.append({
                    "transformed_feature": transformed,
                    "source_feature": _source_feature_name(transformed, self.feature_columns),
                    "original_value": _original_value_for_feature(transformed, row, self.feature_columns),
                    "transformed_value": float(X_dense[0, j]),
                    "shap_value": float(vals[j]),
                })
            return items

        result = {
            "risk": pack(pos),
            "protective": pack(neg),
            "method": "on_demand_linear_shap",
            "scope": "selected enrollment",
            "model_output": "base logistic-regression score before probability calibration",
            "warning": "Local SHAP explains model behavior for this prediction; it does not establish causality.",
        }
        if len(self._local_cache) >= 512:
            self._local_cache.clear()
        self._local_cache[row_index] = result
        return result

    def student(self, row_index: int):
        r = self.progress[self.progress["row_index"] == row_index]
        if r.empty:
            return None
        rec = r.iloc[0]
        try:
            recommendations = json.loads(rec["recommendations_json"])
        except Exception:
            recommendations = []
        payload = {k: _safe(v) for k, v in rec.to_dict().items() if k not in {"recommendations_json", "student_key"}}
        payload["recommendations"] = recommendations
        payload["localExplanation"] = self._local_explain(row_index)
        return payload

    def live_predict(self, row_index: int):
        if row_index not in self.master_by_row.index:
            raise KeyError("row_index not found")
        row = self.master_by_row.loc[row_index]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        b = self.bundle
        X = pd.DataFrame([row[b["feature_columns"]].to_dict()])
        score = _raw_score(b["base_model"], X)
        prob = float(b["calibrator"].predict_proba(score.reshape(-1, 1))[:, 1][0])
        level = _risk_level(prob, b["low_threshold"], b["high_threshold"])
        recs = (
            generate_recommendations(row, b["recommendation_config"])
            if level != "Low"
            else [{
                "factor": "monitor",
                "recommendation": "Continue routine monitoring and reinforce current study habits.",
                "reason": "Risk probability is below the low-risk boundary.",
            }]
        )
        return {
            "row_index": int(row_index),
            "risk_probability": prob,
            "risk_level": level,
            "predicted_at_risk": int(prob >= b["high_threshold"]),
            "recommendations": recs,
            "prediction_point_day": int(b["prediction_point_day"]),
            "risk_level_note": b["risk_level_note"],
        }

    def global_importance(self):
        return [
            {
                "feature": str(r["source_feature"]),
                "importance": float(r["mean_abs_shap"]),
            }
            for _, r in self.global_shap.iterrows()
        ]

    def fairness_rows(self):
        d = self.fairness.copy()
        # Surface a concise, useful subset.
        d = d[d["attribute"].isin(["gender", "age_band", "disability", "imd_band"])]
        return [{k: _safe(v) for k, v in r.items()} for r in d.to_dict("records")]

    def model_comparison(self):
        cols = ["model", "f1_mean", "recall_mean", "roc_auc_mean", "pr_auc_mean", "selection_rank"]
        return [{k: _safe(v) for k, v in r.items()} for r in self.model_cv[cols].to_dict("records")]


STORE = DataStore()


class Handler(BaseHTTPRequestHandler):
    server_version = "StudentRiskDemo/1.2"

    def _common_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )

    def _json(self, data, status=200, head_only=False):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._common_headers()
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _static(self, path: Path, head_only=False):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self._common_headers()
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    @staticmethod
    def _first(q, name, default=None):
        vals = q.get(name)
        return vals[0] if vals else default

    def _required_int(self, q, name):
        value = self._first(q, name)
        if value is None:
            raise ValueError(f"Missing required query parameter: {name}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer query parameter: {name}") from exc

    def _route(self, head_only=False):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._static(STATIC_DIR / "index.html", head_only=head_only)
            if u.path.startswith("/static/"):
                rel = u.path[len("/static/"):]
                path = (STATIC_DIR / rel).resolve()
                if STATIC_DIR.resolve() not in path.parents:
                    return self._json({"error": "Forbidden", "message": "Invalid static path."}, 403, head_only)
                return self._static(path, head_only=head_only)
            if u.path == "/api/health":
                return self._json(STORE.health(), head_only=head_only)
            if u.path == "/api/summary":
                return self._json(STORE.summary(), head_only=head_only)
            if u.path == "/api/students":
                limit_raw = self._first(q, "limit", "80")
                try:
                    limit = int(limit_raw)
                except ValueError as exc:
                    raise ValueError("Invalid integer query parameter: limit") from exc
                return self._json(
                    STORE.students(
                        risk=self._first(q, "risk", "All"),
                        search=self._first(q, "search", ""),
                        limit=limit,
                        sort=self._first(q, "sort", "prob_desc"),
                    ),
                    head_only=head_only,
                )
            if u.path == "/api/student":
                row_index = self._required_int(q, "row_index")
                student = STORE.student(row_index)
                if student is None:
                    return self._json({"error": "NotFound", "message": "row_index not found"}, 404, head_only)
                return self._json(student, head_only=head_only)
            if u.path == "/api/predict":
                return self._json(STORE.live_predict(self._required_int(q, "row_index")), head_only=head_only)
            if u.path == "/api/global-importance":
                return self._json(STORE.global_importance(), head_only=head_only)
            if u.path == "/api/fairness":
                return self._json(STORE.fairness_rows(), head_only=head_only)
            if u.path == "/api/models":
                return self._json(STORE.model_comparison(), head_only=head_only)
        except ValueError as exc:
            return self._json({"error": "BadRequest", "message": str(exc)}, 400, head_only)
        except KeyError as exc:
            return self._json({"error": "NotFound", "message": str(exc)}, 404, head_only)
        except Exception as exc:
            return self._json({"error": type(exc).__name__, "message": str(exc)}, 500, head_only)
        return self._json({"error": "NotFound", "message": "Endpoint not found"}, 404, head_only)

    def do_GET(self):
        return self._route(head_only=False)

    def do_HEAD(self):
        return self._route(head_only=True)

    def log_message(self, fmt, *args):
        print(f"[demo] {self.address_string()} - {fmt % args}")


def main():
    # Local default stays private on 127.0.0.1. Hosting platforms usually provide
    # PORT and require binding to 0.0.0.0, so hosted mode is enabled automatically.
    hosted = bool(os.environ.get("PORT"))
    host = os.environ.get("STUDENT_RISK_HOST", "0.0.0.0" if hosted else "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("STUDENT_RISK_PORT", "8765")))
    server = ThreadingHTTPServer((host, port), Handler)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}"
    print("=" * 62)
    print("Student Academic Risk Dashboard")
    print(f"Open: {url}")
    print("Press Ctrl+C to stop.")
    print("=" * 62)
    if not hosted and os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
