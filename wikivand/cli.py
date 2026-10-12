"""`python -m wikivand <command>`"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .config import CONFIG


def _window(cfg, days: float) -> tuple[float, float, float]:
    """Training window ending at the newest *matured* edit, based on data coverage."""
    from .storage import read_edits, read_reverts

    edits = read_edits(cfg)
    if edits.empty:
        raise SystemExit("no data; run `python -m wikivand ingest` first")
    from .labels import coverage

    coverage_end = coverage(edits, read_reverts(cfg))
    from .dataset import WARMUP_S

    end = coverage_end - cfg.horizon_s
    # rows need a full window of history behind them, like the warmed-up server
    start = max(float(edits.timestamp.min()) + WARMUP_S, end - days * 86400)
    if end <= start:
        hrs = (coverage_end - float(edits.timestamp.min())) / 3600
        raise SystemExit(f"only {hrs:.1f}h of data; need more than the {cfg.label_horizon_h}h label "
                         f"horizon before anything is labelled (or lower WIKIVAND_HORIZON_H)")
    return start, end, coverage_end


def _embedder(name: str | None, cfg):
    from .embed import HashingEmbedder, load_embedder

    if name == "hashing":
        return HashingEmbedder()
    return load_embedder(cfg.embed_model if name is None else name)


def _fmt(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def _parse_iso(ts: str) -> float:
    from .events import parse_ts

    return parse_ts(ts)


def cmd_ingest(a, cfg):
    from .serve import Runner

    Runner(cfg, score=False, fetch_diffs=not a.no_diffs, workers=a.workers, flush_s=a.flush).run(
        since=a.since, duration_s=a.minutes * 60 if a.minutes else None,
        until=_parse_iso(a.until) if a.until else None)


def cmd_serve(a, cfg):
    from .serve import Runner

    Runner(cfg, score=True, workers=a.workers, flush_s=a.flush, model_version=a.model).run(
        duration_s=a.minutes * 60 if a.minutes else None)


def cmd_backfill_diffs(a, cfg):
    """Fetch diffs for past edits. With --neg-per-pos, fetch every matured positive
    plus K random negatives each; dataset.py re-weights so metrics stay honest."""
    import random

    from .dataset import eligible
    from .diffs import Cache, DiffFetcher
    from .labels import build_labels
    from .storage import read_edits, read_reverts

    now = time.time()
    all_edits = read_edits(cfg, now - a.days * 86400, now)
    edits = all_edits[[eligible(cfg, e) for e in all_edits.to_dict("records")]]
    cache = Cache(cfg.cache_db)
    have = cache.get_diffs(cfg.wiki, [int(r) for r in edits.revision_new])
    rng = random.Random(0)
    if a.neg_per_pos:
        lab = build_labels(all_edits, read_reverts(cfg, now - a.days * 86400, now), cfg.horizon_s)
        lab = lab[lab.revision_new.isin(edits.revision_new)]
        lab = lab[~lab.revision_new.isin(have.keys())]
        pos = lab[lab.label == 1].to_dict("records")
        neg = lab[lab.label == 0].to_dict("records")
        n_have_pos = int(edits.revision_new.isin(have.keys()).sum())
        todo = pos + rng.sample(neg, min(len(neg), a.neg_per_pos * len(pos)))
        print(f"positives to fetch: {len(pos)}, negatives: {len(todo) - len(pos)} "
              f"(already cached: {n_have_pos})")
    else:
        todo = [e for e in edits.to_dict("records") if e["revision_new"] not in have]
    if a.limit and len(todo) > a.limit:
        todo = rng.sample(todo, a.limit)
    rng.shuffle(todo)
    print(f"{len(edits)} eligible edits, {len(have)} cached, fetching {len(todo)} "
          f"(~{len(todo) / cfg.api_rps / 60:.0f} min at {cfg.api_rps} req/s)")
    print("note: diffs fetched long after the edit are biased (vandalism gets revision-deleted)")
    f = DiffFetcher(cache, cfg.user_agent, cfg.api_rps)
    with ThreadPoolExecutor(a.workers) as pool:
        st = Counter(d["status"] for d in pool.map(
            lambda e: f.fetch(cfg.wiki, int(e["revision_old"]), int(e["revision_new"])), todo))
    print(dict(st))


def cmd_labels(a, cfg):
    from .labels import build_labels, label_report
    from .storage import read_edits, read_reverts

    edits = read_edits(cfg)
    # label on ALL edits (the reverter is often a bot), then restrict to eligible ones
    lab = build_labels(edits, read_reverts(cfg), cfg.horizon_s)
    lab = lab[lab.namespace.isin(cfg.namespaces) & ~lab.bot]
    rep = label_report(lab, cfg.horizon_s)
    rep["horizon_h"] = cfg.label_horizon_h
    print(json.dumps(rep, indent=2))


def _build(a, cfg):
    from .dataset import build_dataset

    start, end, coverage_end = _window(cfg, a.days)
    emb = _embedder(a.embedder, cfg)
    print(f"window {_fmt(start)} -> {_fmt(end)}  (horizon {cfg.label_horizon_h}h, "
          f"embedder {emb.name if emb else 'none'})")
    df, E = build_dataset(cfg, start, end, emb, coverage_end=coverage_end,
                          require_diff=not a.allow_missing_diffs)
    print(f"{len(df)} labelled edits, {int(df.label.sum())} positives ({df.label.mean():.2%}); "
          f"diff status: {df.diff_status.value_counts().to_dict()}")
    return df, E, emb


def cmd_train(a, cfg):
    from . import model

    df, E, emb = _build(a, cfg)
    if a.ablation:
        print("ablation (test split):")
        print(json.dumps(model.ablation(df, E), indent=2))
    art, rep = model.train(df, E, target_precision=a.target_precision)
    print(json.dumps(rep, indent=2))
    v = model.save(cfg.models_dir, art, {"embedder": emb.name if emb else None,
                                         "horizon_h": cfg.label_horizon_h, "wiki": cfg.wiki})
    if a.promote or model.champion(cfg.models_dir) is None:
        model.promote(cfg.models_dir, v)
        print(f"saved and promoted {v}")
    else:
        print(f"saved {v} (champion unchanged; use `retrain` or --promote)")


def cmd_retrain(a, cfg):
    """Champion/challenger on the most recent matured data."""
    import numpy as np

    from . import model

    df, E, emb = _build(a, cfg)
    challenger, rep = model.train(df, E, target_precision=a.target_precision)
    _, _, te = model.time_split(df)
    y = df.label.to_numpy(int)[te]
    cur = model.champion(cfg.models_dir)
    result = {"challenger": rep["test"]}
    if cur:
        champ, meta = model.load(cfg.models_dir, cur)
        if (meta.get("embedder") or None) != (emb.name if emb else None) and champ["emb_dim"]:
            raise SystemExit(f"champion uses embedder {meta['embedder']}; pass --embedder to match")
        s = model.score(champ, df.iloc[te], E[te])
        w = df["weight"].to_numpy(float)[te]
        result["champion"] = model.metrics(y, s, champ["threshold"], w)
    print(json.dumps(result, indent=2))
    ch_ap = result["challenger"].get("pr_auc", 0)
    cp_ap = result.get("champion", {}).get("pr_auc", -np.inf)
    v = model.save(cfg.models_dir, challenger, {"embedder": emb.name if emb else None,
                                                "horizon_h": cfg.label_horizon_h, "wiki": cfg.wiki,
                                                "challenged": cur})
    if ch_ap >= cp_ap + a.min_gain:
        model.promote(cfg.models_dir, v)
        print(f"promoted {v} (PR-AUC {ch_ap:.3f} vs {cp_ap:.3f})")
    else:
        print(f"kept {cur} (challenger {ch_ap:.3f} vs champion {cp_ap:.3f})")


def cmd_monitor(a, cfg):
    from .monitor import run_monitor

    print(json.dumps(run_monitor(cfg, a.hours), indent=2, default=str))


def cmd_editwars(a, cfg):
    from .dataset import replay
    from .state import OnlineState
    from .storage import read_edits, read_reverts

    now = time.time()
    edits = read_edits(cfg, now - a.days * 86400, now)
    _, alerts = replay(edits, read_reverts(cfg, now - a.days * 86400, now), OnlineState(),
                       keep=lambda e: False)
    for al in alerts:
        print(f"{_fmt(al['ts'])}  {al['title']!r}: {al['reverts_in_window']} reverts, "
              f"mutual={al['mutual_pairs']} 3RR={al['rrr_violators']}")
    print(f"{len(alerts)} edit-war alerts")


def main(argv=None):
    p = argparse.ArgumentParser(prog="wikivand", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="record edits + revert labels from EventStreams")
    s.add_argument("--since", help="replay from ISO time (EventStreams keeps ~7 days)")
    s.add_argument("--no-diffs", action="store_true", help="don't fetch diffs at ingest time")
    s.add_argument("--until", help="stop once both streams reach this ISO time (segmented replay)")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--minutes", type=float)
    s.add_argument("--flush", type=float, default=60)
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("serve", help="ingest + score live edits, report latency")
    s.add_argument("--model", help="model version (default: champion)")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--minutes", type=float)
    s.add_argument("--flush", type=float, default=60)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("backfill-diffs", help="fetch missing diffs for recent edits")
    s.add_argument("--days", type=float, default=7)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, help="random sample size")
    s.add_argument("--neg-per-pos", type=int, help="all matured positives + K negatives each")
    s.set_defaults(fn=cmd_backfill_diffs)

    sub.add_parser("labels", help="label maturity, base rate, time-to-revert").set_defaults(fn=cmd_labels)

    for name, fn in (("train", cmd_train), ("retrain", cmd_retrain)):
        s = sub.add_parser(name, help="train a model" if name == "train" else "champion/challenger retrain")
        s.add_argument("--days", type=float, default=7, help="training window length")
        s.add_argument("--embedder", help="'hashing', 'none' or a sentence-transformers model")
        s.add_argument("--target-precision", type=float, default=0.5,
                       help="alert threshold = lowest score reaching this precision on val")
        s.add_argument("--allow-missing-diffs", action="store_true",
                       help="train on rows without diffs (metadata/state-only experiments)")
        if name == "train":
            s.add_argument("--ablation", action="store_true")
            s.add_argument("--promote", action="store_true")
        else:
            s.add_argument("--min-gain", type=float, default=0.005)
        s.set_defaults(fn=fn)

    s = sub.add_parser("monitor", help="latency, drift, delayed performance")
    s.add_argument("--hours", type=float, default=24)
    s.set_defaults(fn=cmd_monitor)

    s = sub.add_parser("editwars", help="replay logged data through the edit-war detector")
    s.add_argument("--days", type=float, default=7)
    s.set_defaults(fn=cmd_editwars)

    a = p.parse_args(argv)
    # SIGTERM (kill, systemd, docker stop) should flush buffers like Ctrl-C does
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    a.fn(a, CONFIG)
