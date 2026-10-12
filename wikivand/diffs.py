"""Fetch and cache text diffs (and embeddings) per revision.

Diffs must be fetched close to edit time: vandalism is often revision-deleted
later, so a backfill would see "missing content" mostly on positives (label leak).
`status` records why a diff is missing so that bias is measurable.
"""
from __future__ import annotations

import html
import re
import sqlite3
import threading
import time
from pathlib import Path

import httpx
import numpy as np

_SPECIAL_DOMAINS = {
    "commonswiki": "commons.wikimedia.org",
    "wikidatawiki": "www.wikidata.org",
    "metawiki": "meta.wikimedia.org",
}
_ADDED = re.compile(r'<td class="diff-addedline[^"]*">(.*?)</td>', re.S)
_DELETED = re.compile(r'<td class="diff-deletedline[^"]*">(.*?)</td>', re.S)
_INS = re.compile(r"<ins[^>]*>(.*?)</ins>", re.S)
_DEL = re.compile(r"<del[^>]*>(.*?)</del>", re.S)
_TAG = re.compile(r"<[^>]+>")


def wiki_domain(wiki: str) -> str:
    if wiki in _SPECIAL_DOMAINS:
        return _SPECIAL_DOMAINS[wiki]
    return f"{wiki.removesuffix('wiki').replace('_', '-')}.wikipedia.org"


def _cells_text(body: str, cell: re.Pattern, inline: re.Pattern) -> str:
    out = []
    for c in cell.findall(body):
        parts = inline.findall(c)  # changed line: keep only the changed spans
        text = " ".join(parts) if parts else c
        out.append(html.unescape(_TAG.sub("", text)))
    return "\n".join(s for s in out if s.strip())


def parse_diff_html(body: str) -> tuple[str, str]:
    """Return (added_text, removed_text) from a MediaWiki diff table."""
    return _cells_text(body, _ADDED, _INS), _cells_text(body, _DELETED, _DEL)


class Cache:
    """SQLite cache for diffs and embeddings. Safe to share across threads."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS diffs(wiki TEXT, rev_id INTEGER, status TEXT,
                added TEXT, removed TEXT, fetched_ts REAL, PRIMARY KEY (wiki, rev_id));
            CREATE TABLE IF NOT EXISTS embeddings(wiki TEXT, rev_id INTEGER, model TEXT,
                vec BLOB, PRIMARY KEY (wiki, rev_id, model));
            """
        )

    def get_diff(self, wiki: str, rev_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT status, added, removed FROM diffs WHERE wiki=? AND rev_id=?", (wiki, rev_id)
            ).fetchone()
        return None if row is None else {"status": row[0], "added": row[1], "removed": row[2]}

    def get_diffs(self, wiki: str, rev_ids: list[int]) -> dict[int, dict]:
        out = {}
        with self._lock:
            for i in range(0, len(rev_ids), 900):
                chunk = rev_ids[i : i + 900]
                q = f"SELECT rev_id, status, added, removed FROM diffs WHERE wiki=? AND rev_id IN ({','.join('?' * len(chunk))})"
                for r, s, a, d in self._db.execute(q, (wiki, *chunk)):
                    out[r] = {"status": s, "added": a, "removed": d}
        return out

    def put_diff(self, wiki: str, rev_id: int, d: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO diffs VALUES (?,?,?,?,?,?)",
                (wiki, rev_id, d["status"], d["added"], d["removed"], time.time()),
            )

    def get_embeddings(self, wiki: str, model: str, rev_ids: list[int]) -> dict[int, np.ndarray]:
        out = {}
        with self._lock:
            for i in range(0, len(rev_ids), 900):
                chunk = rev_ids[i : i + 900]
                q = f"SELECT rev_id, vec FROM embeddings WHERE wiki=? AND model=? AND rev_id IN ({','.join('?' * len(chunk))})"
                for r, v in self._db.execute(q, (wiki, model, *chunk)):
                    out[r] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_embeddings(self, wiki: str, model: str, items: dict[int, np.ndarray]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
                [(wiki, model, r, v.astype(np.float32).tobytes()) for r, v in items.items()],
            )


class RateLimiter:
    """Process-wide request pacing plus a shared back-off when the API says 429."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.next_ok = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_ok)
            self.next_ok = t + self.interval
        if t > now:
            time.sleep(t - now)

    def back_off(self, seconds: float) -> None:
        with self.lock:
            self.next_ok = max(self.next_ok, time.monotonic() + seconds)


class DiffFetcher:
    def __init__(self, cache: Cache, user_agent: str, rps: float = 3.0, retries: int = 2):
        self.cache = cache
        self.client = httpx.Client(headers={"User-Agent": user_agent}, timeout=10.0)
        self.limiter = RateLimiter(rps)
        self.retries = retries

    def _get(self, wiki: str, params: dict) -> httpx.Response:
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            r = self.client.get(f"https://{wiki_domain(wiki)}/w/api.php", params=params)
            if r.status_code != 429 or attempt == self.retries:
                return r
            self.limiter.back_off(float(r.headers.get("retry-after", 5)))
        return r

    def fetch(self, wiki: str, old: int, new: int) -> dict:
        cached = self.cache.get_diff(wiki, new)
        if cached is not None:
            return cached
        if not old:
            d = {"status": "no_parent", "added": "", "removed": ""}
        else:
            try:
                r = self._get(wiki, {"action": "compare", "fromrev": old, "torev": new, "prop": "diff",
                                     "format": "json", "formatversion": 2, "maxlag": 5})
                r.raise_for_status()
                js = r.json()
                if "error" in js:
                    d = {"status": f"api:{js['error'].get('code')}", "added": "", "removed": ""}
                else:
                    added, removed = parse_diff_html(js["compare"].get("body", ""))
                    d = {"status": "ok", "added": added, "removed": removed}
            except (httpx.HTTPError, ValueError) as e:
                # transient: don't cache, so a later backfill can retry
                return {"status": f"http:{type(e).__name__}", "added": "", "removed": ""}
        self.cache.put_diff(wiki, new, d)
        return d
