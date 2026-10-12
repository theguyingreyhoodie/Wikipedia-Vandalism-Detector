"""Training, evaluation and a tiny file-based model registry with champion/challenger."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

from .features import FEATURES, META_FEATURES, STATE_FEATURES

EMB_COMPONENTS = 24


def make_pipeline(n_tab: int, n_emb: int, n_rows: int) -> Pipeline:
    parts = [("tab", "passthrough", list(range(n_tab)))]
    if n_emb:
        k = max(1, min(EMB_COMPONENTS, n_emb, n_rows - 1))
        parts.append(("emb", PCA(n_components=k, random_state=0), list(range(n_tab, n_tab + n_emb))))
    return Pipeline([
        ("cols", ColumnTransformer(parts)),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, class_weight="balanced", random_state=0)),
    ])


def matrix(df: pd.DataFrame, emb: np.ndarray, features: list[str]) -> np.ndarray:
    tab = df[features].to_numpy(np.float32)
    if tab.shape[1] != len(features):  # duplicate column names would silently widen X
        raise ValueError(f"expected {len(features)} feature columns, got {tab.shape[1]}")
    return np.hstack([tab, emb]).astype(np.float32)


def metrics(y: np.ndarray, s: np.ndarray, threshold: float | None = None,
            w: np.ndarray | None = None) -> dict:
    """Ranking + operating-point metrics. `w` = importance weights (see dataset.py)."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "base_rate": float(np.average(y, weights=w)) if len(y) else None}
    if len(y) == 0 or y.min() == y.max():
        return out  # AP/AUC undefined with one class
    out["pr_auc"] = float(average_precision_score(y, s, sample_weight=w))
    out["roc_auc"] = float(roc_auc_score(y, s, sample_weight=w))
    order = np.argsort(-s)
    top = order[: max(1, int(np.searchsorted(np.cumsum(w[order]), 0.01 * w.sum())) + 1)]
    out["precision_at_top1pct"] = float(np.average(y[top], weights=w[top]))
    p, r, _ = precision_recall_curve(y, s, sample_weight=w)
    for target in (0.5, 0.8, 0.9):
        ok = p >= target
        out[f"recall_at_p{int(target * 100)}"] = float(r[ok].max()) if ok.any() else 0.0
    if threshold is not None:
        pred = s >= threshold
        tp = float(w[pred & (y == 1)].sum())
        out["threshold"] = float(threshold)
        out["precision"] = tp / max(float(w[pred].sum()), 1e-9) if pred.any() else 0.0
        out["recall"] = tp / float(w[y == 1].sum())
        out["alerts_per_1k_edits"] = 1000 * float(np.average(pred, weights=w))
    return out


def pick_threshold(y: np.ndarray, s: np.ndarray, target_precision: float,
                   w: np.ndarray | None = None) -> float:
    """Lowest threshold reaching target precision on validation; else max-F1."""
    if y.min() == y.max():
        return 0.5
    p, r, t = precision_recall_curve(y, s, sample_weight=w)
    p, r = p[:-1], r[:-1]
    ok = np.where(p >= target_precision)[0]
    if len(ok):
        return float(t[ok[0]])
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    return float(t[np.argmax(f1)])


def time_split(df: pd.DataFrame, fracs=(0.6, 0.2)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split. Never shuffle: tomorrow's vandals aren't in today's data."""
    order = np.argsort(df.timestamp.to_numpy(), kind="stable")
    a = int(len(df) * fracs[0])
    b = int(len(df) * (fracs[0] + fracs[1]))
    return order[:a], order[a:b], order[b:]


def train(df: pd.DataFrame, emb: np.ndarray, *, features=FEATURES, target_precision=0.5,
          use_emb=True) -> tuple[dict, dict]:
    tr, va, te = time_split(df)
    E = emb if use_emb else emb[:, :0]
    X = matrix(df, E, features)
    y = df.label.to_numpy(int)
    w = df["weight"].to_numpy(float) if "weight" in df else np.ones(len(df))
    if y[tr].min() == y[tr].max():
        raise SystemExit("training split has a single class; collect more data")
    pipe = make_pipeline(len(features), E.shape[1], len(tr)).fit(X[tr], y[tr])
    ref = np.random.default_rng(0).choice(tr, size=5000, p=w[tr] / w[tr].sum())
    s_va = pipe.predict_proba(X[va])[:, 1]
    s_te = pipe.predict_proba(X[te])[:, 1]
    thr = pick_threshold(y[va], s_va, target_precision, w[va])
    report = {
        "train": {"n": int(len(tr)), "positives": int(y[tr].sum())},
        "val": metrics(y[va], s_va, thr, w[va]),
        "test": metrics(y[te], s_te, thr, w[te]),
        "window": [float(df.timestamp.min()), float(df.timestamp.max())],
    }
    artifact = {
        "pipeline": pipe, "features": list(features), "emb_dim": int(E.shape[1]),
        "threshold": thr, "report": report,
        # Drift reference: training rows resampled by importance weight, so it
        # looks like live traffic rather than the positive-enriched sample.
        "reference": df.iloc[ref][list(features)],
        "reference_scores": pipe.predict_proba(X[ref])[:, 1],
    }
    return artifact, report


def ablation(df: pd.DataFrame, emb: np.ndarray) -> dict:
    """How much does each feature family buy? Same split, same model."""
    out = {}
    for name, feats, use_emb in [
        ("metadata", META_FEATURES, False),
        ("+state", META_FEATURES + STATE_FEATURES, False),
        ("+diff_text", FEATURES, False),
        ("+embedding", FEATURES, True),
    ]:
        if use_emb and emb.shape[1] == 0:
            continue
        _, rep = train(df, emb, features=feats, use_emb=use_emb)
        out[name] = {k: rep["test"].get(k) for k in ("pr_auc", "roc_auc", "recall_at_p80")}
    return out


def score(artifact: dict, df: pd.DataFrame, emb: np.ndarray) -> np.ndarray:
    E = emb[:, : artifact["emb_dim"]] if artifact["emb_dim"] else emb[:, :0]
    return artifact["pipeline"].predict_proba(matrix(df, E, artifact["features"]))[:, 1]


# ---- registry --------------------------------------------------------------

def save(models_dir: Path, artifact: dict, meta: dict) -> str:
    version = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    d = models_dir / version
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, d / "model.joblib")
    (d / "meta.json").write_text(json.dumps({**meta, "version": version, "report": artifact["report"]},
                                            indent=2, default=str))
    return version


def promote(models_dir: Path, version: str) -> None:
    (models_dir / "CHAMPION").write_text(version)


def champion(models_dir: Path) -> str | None:
    p = models_dir / "CHAMPION"
    return p.read_text().strip() if p.exists() else None


def load(models_dir: Path, version: str | None = None) -> tuple[dict, dict]:
    version = version or champion(models_dir)
    if not version:
        raise SystemExit("no champion model; run `wikivand train` first")
    d = models_dir / version
    return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text())
