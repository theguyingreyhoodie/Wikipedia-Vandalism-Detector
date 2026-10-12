"""Delayed labels from revert events.

label_status per edit:
  pos          reverted by someone else within the horizon
  neg          horizon has fully elapsed (relative to data coverage) with no revert
               by someone else inside it
  pending      too recent to know; must NOT be used for training or evaluation.
               Keeping early positives while dropping pending negatives would
               inflate the positive rate of the newest data.
  self_revert  the editor reverted themselves; neither vandalism nor clean
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def attach_reverter(edits: pd.DataFrame, reverts: pd.DataFrame, slack_s: float = 10.0) -> pd.DataFrame:
    """Guess who made each revert: the latest editor of the page at revert time.

    tags-change events don't name the reverting edit, but it is the edit to the
    same page that landed just before the tag was applied.
    """
    if reverts.empty:
        return reverts.assign(reverter=pd.Series(dtype=str))
    left = reverts.assign(_t=reverts.revert_ts + slack_s).sort_values("_t")
    right = edits[["wiki", "title", "timestamp", "user"]].rename(columns={"user": "reverter"})
    right = right.assign(_t=right.timestamp.astype(float), _rt=right.timestamp.astype(float))
    right = right.drop(columns="timestamp").sort_values("_t")
    out = pd.merge_asof(left, right, on="_t", by=["wiki", "title"], direction="backward")
    # The reverting edit must come after the reverted one; otherwise we never saw it.
    out.loc[~(out._rt > out.rev_ts), "reverter"] = None
    return out.drop(columns=["_t", "_rt"])


def coverage(edits: pd.DataFrame, reverts: pd.DataFrame) -> float:
    """Event time up to which BOTH streams have been observed.

    Uses event time, not ingestion time: during a `--since` replay, ingestion is
    "now" while the revert stream may still be hours in the past.
    """
    if edits.empty or reverts.empty:
        return 0.0
    return min(float(edits.timestamp.max()), float(reverts.revert_ts.max()))


def build_labels(edits: pd.DataFrame, reverts: pd.DataFrame, horizon_s: float,
                 coverage_end: float | None = None) -> pd.DataFrame:
    """Return edits with label_status, label (1/0/NaN) and time_to_revert."""
    e = edits.copy()
    if coverage_end is None:
        coverage_end = coverage(edits, reverts)
    if len(reverts):
        r = attach_reverter(edits, reverts)
        r = r.merge(edits[["wiki", "revision_new", "user"]], left_on=["wiki", "rev_id"],
                    right_on=["wiki", "revision_new"], how="left")
        r["self_revert"] = r.reverter.notna() & (r.reverter == r.user)
        r = (r.sort_values("revert_ts").groupby(["wiki", "rev_id"], as_index=False)
             .agg(revert_ts=("revert_ts", "first"), self_revert=("self_revert", "first"),
                  reverter=("reverter", "first")))
        e = e.merge(r, left_on=["wiki", "revision_new"], right_on=["wiki", "rev_id"], how="left")
        e = e.drop(columns="rev_id")
    else:
        e["revert_ts"], e["self_revert"], e["reverter"] = np.nan, False, None
    e["self_revert"] = e.self_revert.fillna(False).astype(bool)
    e["time_to_revert"] = e.revert_ts - e.timestamp

    reverted_in_h = e.time_to_revert.notna() & (e.time_to_revert <= horizon_s)
    matured = e.timestamp + horizon_s <= coverage_end
    status = np.select(
        [reverted_in_h & e.self_revert, ~matured, reverted_in_h],
        ["self_revert", "pending", "pos"],
        default="neg",
    )
    e["label_status"] = status
    e["label"] = np.where(status == "pos", 1.0, np.where(status == "neg", 0.0, np.nan))
    return e


def label_report(labeled: pd.DataFrame, horizon_s: float) -> dict:
    ttr = labeled.time_to_revert.dropna()
    ttr = ttr[ttr >= 0]
    counts = labeled.label_status.value_counts().to_dict()
    known = labeled.label.dropna()
    q = ttr.quantile([0.5, 0.9, 0.99]).to_dict() if len(ttr) else {}
    return {
        "edits": int(len(labeled)),
        "status_counts": {k: int(v) for k, v in counts.items()},
        "positive_rate_matured": float(known.mean()) if len(known) else None,
        "reverts_seen": int(len(ttr)),
        "time_to_revert_min": {f"p{int(k * 100)}": round(v / 60, 1) for k, v in q.items()},
        "share_of_reverts_within_horizon": float((ttr <= horizon_s).mean()) if len(ttr) else None,
    }
