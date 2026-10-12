"""One process for the live path: ingest both streams, persist them, and (optionally)
score every eligible edit with per-stage latency accounting."""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .dataset import WARMUP_S, eligible, replay
from .diffs import Cache, DiffFetcher
from .embed import load_embedder
from .events import EDIT_STREAM, TAGS_STREAM, edit_row, revert_row
from .features import embed_text, featurize
from .model import load, matrix
from .sse import iter_merged
from .state import OnlineState
from .storage import read_edits, read_reverts, write_batch

log = logging.getLogger(__name__)
STAGES = ("queue", "diff", "features", "embed", "predict")


class Runner:
    def __init__(self, cfg, *, score: bool = True, fetch_diffs: bool = True, workers: int = 8,
                 flush_s: float = 60.0, model_version: str | None = None):
        self.cfg = cfg
        self.score = score
        self.fetch_diffs = fetch_diffs or score
        self.flush_s = flush_s
        self.cache = Cache(cfg.cache_db)
        self.fetcher = DiffFetcher(self.cache, cfg.user_agent, cfg.api_rps)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.max_pending = workers * 50
        self.pending = 0
        self.lock = threading.Lock()
        self.state = OnlineState()
        self.buf = {"edits": [], "reverts": [], "predictions": []}
        self.stats = Counter()
        self.latency: deque[float] = deque(maxlen=5000)

        self.artifact = self.embedder = None
        if score:
            self.artifact, self.meta = load(cfg.models_dir, model_version)
            self.version = self.meta["version"]
            self.embedder = load_embedder(self.meta.get("embedder")) if self.artifact["emb_dim"] else None
            if self.embedder and self.embedder.name != self.meta.get("embedder"):
                raise SystemExit(f"model needs embedder {self.meta['embedder']!r}, got {self.embedder.name!r}")
            self.threshold = self.artifact["threshold"]
            log.info("serving model %s (threshold %.3f)", self.version, self.threshold)

    def warm_up(self) -> None:
        """Rebuild point-in-time state from the last day of logged events."""
        end = time.time()
        edits = read_edits(self.cfg, end - WARMUP_S, end)
        reverts = read_reverts(self.cfg, end - WARMUP_S, end)
        if len(edits):
            replay(edits, reverts, self.state, keep=lambda e: False)
        log.info("state warmed with %d edits, %d reverts", len(edits), len(reverts))

    # ---- scoring -----------------------------------------------------------

    def _score(self, e: dict, snap: dict, t_recv: float) -> None:
        try:
            t0 = time.perf_counter()
            queue_s = time.time() - t_recv
            diff = self.fetcher.fetch(self.cfg.wiki, e["revision_old"], e["revision_new"])
            t1 = time.perf_counter()
            feats = featurize(e, snap, diff)
            t2 = time.perf_counter()
            if self.embedder:
                emb = self.embedder.encode([embed_text(diff)])
            else:
                emb = np.zeros((1, 0), np.float32)
            t3 = time.perf_counter()
            s = float(self.artifact["pipeline"].predict_proba(
                matrix(pd.DataFrame([feats]), emb, self.artifact["features"]))[0, 1])
            t4 = time.perf_counter()
            lat = {"queue": queue_s * 1000, "diff": (t1 - t0) * 1000, "features": (t2 - t1) * 1000,
                   "embed": (t3 - t2) * 1000, "predict": (t4 - t3) * 1000}
            total = sum(lat.values())
            row = {
                "wiki": e["wiki"], "revision_new": e["revision_new"], "timestamp": e["timestamp"],
                "title": e["title"], "user": e["user"], "score": s, "alert": s >= self.threshold,
                "model_version": self.version, "diff_status": diff["status"],
                # end-to-end: edit saved -> score ready (EventStreams delay included)
                "event_lag_ms": (t_recv - e["timestamp"]) * 1000,
                "latency_ms": total, **{f"lat_{k}_ms": v for k, v in lat.items()},
                **{f"f_{k}": v for k, v in feats.items()},
            }
            with self.lock:
                self.buf["predictions"].append(row)
                self.latency.append(total)
                self.stats["scored"] += 1
                if row["alert"]:
                    self.stats["alerts"] += 1
            if row["alert"]:
                print(f"🚩 {s:.2f}  {e['title']!r} by {e['user']}  "
                      f"https://{self._domain()}/w/index.php?diff={e['revision_new']}  ({total:.0f} ms)",
                      flush=True)
        except Exception:
            log.exception("scoring failed for rev %s", e.get("revision_new"))
        finally:
            with self.lock:
                self.pending -= 1

    def _prefetch(self, e: dict) -> None:
        try:
            self.fetcher.fetch(self.cfg.wiki, e["revision_old"], e["revision_new"])
        finally:
            with self.lock:
                self.pending -= 1

    def _domain(self) -> str:
        from .diffs import wiki_domain
        return wiki_domain(self.cfg.wiki)

    # ---- main loop ---------------------------------------------------------

    def _on_edit(self, e: dict) -> None:
        self.buf["edits"].append(e)
        t_recv = time.time()
        want = eligible(self.cfg, e) and (self.score or self.fetch_diffs)
        snap = self.state.snapshot(e) if want else None
        alert = self.state.observe_edit(e)
        if alert:
            self._editwar(alert)
        if not want:
            return
        with self.lock:
            if self.pending >= self.max_pending:  # backpressure: shed load, but count it
                self.stats["dropped"] += 1
                return
            self.pending += 1
        if self.score:
            self.pool.submit(self._score, e, snap, t_recv)
        else:
            self.pool.submit(self._prefetch, e)

    def _editwar(self, alert: dict) -> None:
        self.stats["editwars"] += 1
        print(f"⚔️  edit war on {alert['title']!r}: {alert['reverts_in_window']} reverts/24h, "
              f"mutual={alert['mutual_pairs']} 3RR={alert['rrr_violators']}", flush=True)
        path = self.cfg.data_dir / "alerts" / "editwars.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(alert) + "\n")

    def flush(self) -> None:
        with self.lock:
            bufs, self.buf = self.buf, {"edits": [], "reverts": [], "predictions": []}
        write_batch(self.cfg.raw_dir, "edits", bufs["edits"], "timestamp")
        write_batch(self.cfg.reverts_dir, "reverts", bufs["reverts"], "revert_ts")
        write_batch(self.cfg.predictions_dir, "predictions", bufs["predictions"], "timestamp")

    def _status(self, started: float) -> None:
        with self.lock:
            lat = np.array(self.latency)
            s = dict(self.stats)
        el = time.time() - started
        line = (f"[{el / 60:5.1f} min] edits={s.get('edits', 0)} reverts={s.get('reverts', 0)} "
                f"scored={s.get('scored', 0)} alerts={s.get('alerts', 0)} dropped={s.get('dropped', 0)} "
                f"editwars={s.get('editwars', 0)}")
        if len(lat):
            p50, p95, p99 = np.percentile(lat, [50, 95, 99])
            line += f" | latency p50={p50:.0f}ms p95={p95:.0f}ms p99={p99:.0f}ms"
        print(line, flush=True)

    def run(self, since: str | None = None, duration_s: float | None = None) -> None:
        if not since:
            self.warm_up()
        started = last_flush = time.time()
        events = iter_merged([EDIT_STREAM, TAGS_STREAM], base_url=self.cfg.stream_url,
                             user_agent=self.cfg.user_agent, since=since,
                             contains=f'"{self.cfg.wiki}"'.encode())
        try:
            for ev in events:
                stream = (ev.get("meta") or {}).get("stream")
                if stream == EDIT_STREAM and ev.get("wiki") == self.cfg.wiki:
                    e = edit_row(ev)
                    if e:
                        self.stats["edits"] += 1
                        self._on_edit(e)
                elif stream == TAGS_STREAM and ev.get("database") == self.cfg.wiki:
                    r = revert_row(ev)
                    if r:
                        self.stats["reverts"] += 1
                        with self.lock:
                            self.buf["reverts"].append(r)
                        self.state.observe_revert(r)
                now = time.time()
                if now - last_flush >= self.flush_s:
                    self.flush()
                    self._status(started)
                    last_flush = now
                if duration_s and now - started >= duration_s:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.pool.shutdown(wait=True)
            self.flush()
            self._status(started)
