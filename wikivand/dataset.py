"""Build a training/eval frame by replaying the event log through OnlineState."""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from .diffs import Cache
from .events import is_batch_tool
from .features import FEATURES, embed_text, featurize
from .labels import build_labels
from .state import WINDOW, OnlineState
from .storage import read_edits, read_reverts

WARMUP_S = WINDOW  # history replayed before the first training row / at server start


def replay(edits: pd.DataFrame, reverts: pd.DataFrame, state: OnlineState, keep=lambda e: True):
    """Feed edits and reverts to `state` in time order; return snapshots for kept edits."""
    edit_recs = edits.to_dict("records")
    rev_recs = reverts.to_dict("records") if len(reverts) else []
    timeline = heapq.merge(
        ((float(e["timestamp"]), 0, i, e) for i, e in enumerate(edit_recs)),
        ((float(r["revert_ts"]), 1, i, r) for i, r in enumerate(rev_recs)),
    )
    snaps, alerts = {}, []
    for _, kind, _, rec in timeline:
        if kind == 0:
            if keep(rec):
                snaps[rec["revision_new"]] = state.snapshot(rec)
            alert = state.observe_edit(rec)
            if alert:
                alerts.append(alert)
        else:
            state.observe_revert(rec)
    return snaps, alerts


def eligible(cfg, e: dict) -> bool:
    """Edits we score: configured namespaces, no bots, no semi-automated batch tools."""
    return (int(e["namespace"]) in cfg.namespaces and not bool(e["bot"])
            and not is_batch_tool(e.get("comment") or ""))


USABLE_DIFF = ("ok", "no_parent")


def build_dataset(cfg, start: float, end: float, embedder=None, cache: Cache | None = None,
                  coverage_end: float | None = None,
                  require_diff: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows = eligible, matured edits in [start, end). Returns (frame, embeddings).

    With `require_diff`, rows whose diff we never saw are dropped: live serving
    always has the diff, so training on diff-less rows would be train/serve skew.
    API errors (often revision-deletion of vandalism) are dropped too, because
    that status reflects what happened *after* the edit.
    """
    edits = read_edits(cfg, start - WARMUP_S, end + cfg.horizon_s + 3600)
    reverts = read_reverts(cfg, start - WARMUP_S, end + cfg.horizon_s + 3600)
    if edits.empty:
        raise SystemExit("no edits in range; run `wikivand ingest` first")

    labeled = build_labels(edits, reverts, cfg.horizon_s, coverage_end)
    # Only replay up to `end` so state never sees the future.
    past_edits = edits[edits.timestamp < end]
    past_reverts = reverts[reverts.revert_ts < end] if len(reverts) else reverts
    snaps, _ = replay(past_edits, past_reverts, OnlineState(),
                      keep=lambda e: e["timestamp"] >= start and eligible(cfg, e))

    df = labeled[labeled.revision_new.isin(snaps.keys())]
    df = df[df.label.notna()].reset_index(drop=True)
    n_all = df.groupby([(df.timestamp // 3600).astype(int), df.label]).size()
    cache = cache or Cache(cfg.cache_db)
    diffs = cache.get_diffs(cfg.wiki, [int(r) for r in df.revision_new])
    recs = df.to_dict("records")
    feats = [featurize(e, snaps[e["revision_new"]], diffs.get(e["revision_new"])) for e in recs]
    X = pd.DataFrame(feats, columns=FEATURES)
    # feature columns win over same-named raw columns (e.g. `namespace`)
    df = pd.concat([df.drop(columns=[c for c in FEATURES if c in df]), X], axis=1)
    df["diff_status"] = [diffs.get(r, {}).get("status", "missing") for r in df.revision_new]
    if require_diff:
        usable = df.diff_status.isin(USABLE_DIFF)
        dropped = df[~usable].groupby("diff_status").label.agg(["count", "mean"])
        if len(dropped):
            print("dropped rows without a usable diff (count, positive rate):")
            print(dropped.round(3).to_string())
        df = df[usable].reset_index(drop=True)
    # Importance weights undo label-dependent diff sampling (all positives + a
    # fraction of negatives, possibly at different rates in different backfill
    # passes): weight = rows in (hour, class) / rows kept in (hour, class).
    # Metrics and thresholds use them, so precision reflects the real base rate.
    keys = [(df.timestamp // 3600).astype(int), df.label]
    df["weight"] = (n_all.reindex(pd.MultiIndex.from_arrays(keys)).to_numpy()
                    / df.groupby(keys).label.transform("size").to_numpy())

    emb = embed_frame(df, diffs, embedder, cache, cfg.wiki)
    return df, emb


def embed_frame(df, diffs, embedder, cache: Cache, wiki: str) -> np.ndarray:
    if embedder is None:
        return np.zeros((len(df), 0), dtype=np.float32)
    revs = [int(r) for r in df.revision_new]
    have = cache.get_embeddings(wiki, embedder.name, revs)
    todo = [r for r in revs if r not in have]
    if todo:
        vecs = embedder.encode([embed_text(diffs.get(r)) for r in todo])
        new = dict(zip(todo, vecs))
        # Only cache embeddings of real diffs; a diff may still arrive later.
        cache.put_embeddings(wiki, embedder.name,
                             {r: v for r, v in new.items() if diffs.get(r, {}).get("status") == "ok"})
        have.update(new)
    return np.stack([have[r] for r in revs]) if revs else np.zeros((0, embedder.dim), np.float32)
