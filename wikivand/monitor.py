"""Monitoring that respects delayed labels.

  * latency       per-stage percentiles from the prediction log
  * data drift    PSI of live features and scores vs the model's training reference;
                  available immediately, no labels needed
  * label drift   base rate and time-to-revert over time
  * performance   PR-AUC on predictions whose labels have *matured*, bucketed by
                  hour. This lags reality by the label horizon, which is why the
                  label-free signals above exist.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .labels import build_labels, label_report
from .model import load, metrics
from .storage import read_edits, read_predictions, read_reverts


CYCLIC = {"hour_utc"}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index on reference quantile bins. >0.25 = major shift."""
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 50 or len(cur) < 50:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: compare share of the dominant value
        v = np.median(ref)
        p, q = np.mean(ref == v), np.mean(cur == v)
        a, b = np.clip([p, 1 - p], 1e-4, None), np.clip([q, 1 - q], 1e-4, None)
        return float(np.sum((b - a) * np.log(b / a)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    b = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def latency_report(pred: pd.DataFrame) -> dict:
    cols = ["latency_ms"] + [c for c in pred.columns if c.startswith("lat_")] + ["event_lag_ms"]
    return {c.removesuffix("_ms"): {f"p{q}": round(float(np.percentile(pred[c].dropna(), q)), 1)
                                    for q in (50, 95, 99)} for c in cols if pred[c].notna().any()}


def drift_report(artifact: dict, pred: pd.DataFrame) -> dict:
    ref = artifact["reference"]
    out = {"score_psi": psi(artifact["reference_scores"], pred.score.to_numpy(float))}
    feat = {}
    for f in artifact["features"]:
        if f in CYCLIC:  # a few hours of traffic can't match a full-day reference
            continue
        col = f"f_{f}"
        if col in pred:
            feat[f] = psi(ref[f].to_numpy(float), pred[col].to_numpy(float))
    out["feature_psi_top"] = dict(sorted(((k, round(v, 3)) for k, v in feat.items() if v == v),
                                         key=lambda kv: -kv[1])[:8])
    out["features_major_shift"] = [k for k, v in feat.items() if v > 0.25]
    return out


def delayed_performance(cfg, pred: pd.DataFrame, threshold: float, bucket: str = "6h") -> dict:
    lo, hi = float(pred.timestamp.min()), float(pred.timestamp.max())
    edits = read_edits(cfg, lo - 3600, hi + cfg.horizon_s + 3600)
    reverts = read_reverts(cfg, lo - 3600, hi + cfg.horizon_s + 3600)
    lab = build_labels(edits, reverts, cfg.horizon_s)
    j = pred.merge(lab[["wiki", "revision_new", "label", "label_status"]], on=["wiki", "revision_new"])
    out = {"matured": int(j.label.notna().sum()), "pending": int((j.label_status == "pending").sum())}
    m = j[j.label.notna()]
    if len(m):
        out["overall"] = metrics(m.label.to_numpy(int), m.score.to_numpy(float), threshold)
        m = m.assign(bucket=pd.to_datetime(m.timestamp, unit="s", utc=True).dt.floor(bucket))
        out["by_bucket"] = {
            str(b): {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in metrics(g.label.to_numpy(int), g.score.to_numpy(float)).items()
                     if k in ("n", "base_rate", "pr_auc")}
            for b, g in m.groupby("bucket")
        }
    return out


def run_monitor(cfg, hours: float = 24.0) -> dict:
    now = time.time()
    report: dict = {}
    edits = read_edits(cfg, now - hours * 3600 - cfg.horizon_s, now)
    reverts = read_reverts(cfg, now - hours * 3600 - cfg.horizon_s, now)
    if len(edits):
        lab = build_labels(edits, reverts, cfg.horizon_s)
        lab = lab[lab.namespace.isin(cfg.namespaces) & ~lab.bot]
        report["labels"] = label_report(lab, cfg.horizon_s)

    pred = read_predictions(cfg, now - hours * 3600, now)
    if pred.empty:
        report["note"] = "no predictions logged yet; run `wikivand serve`"
        return report
    artifact, meta = load(cfg.models_dir)
    pred_cur = pred[pred.model_version == meta["version"]]
    report["model"] = meta["version"]
    report["latency_ms"] = latency_report(pred)
    report["alert_rate_per_1k"] = round(1000 * float(pred_cur.alert.mean()), 2) if len(pred_cur) else None
    if len(pred_cur):
        report["drift"] = drift_report(artifact, pred_cur)
        report["performance"] = delayed_performance(cfg, pred_cur, artifact["threshold"])

    reasons = []
    d = report.get("drift", {})
    if d.get("score_psi", 0) > 0.25:
        reasons.append(f"score PSI {d['score_psi']:.2f} > 0.25")
    if d.get("features_major_shift"):
        reasons.append(f"feature shift: {', '.join(d['features_major_shift'][:5])}")
    live_ap = report.get("performance", {}).get("overall", {}).get("pr_auc")
    test_ap = meta["report"]["test"].get("pr_auc")
    if live_ap is not None and test_ap and live_ap < 0.8 * test_ap:
        reasons.append(f"matured PR-AUC {live_ap:.3f} < 80% of offline {test_ap:.3f}")
    report["retrain_recommended"] = reasons
    return report
