"""Point-in-time feature state, shared by offline replay and live serving.

Training replays the parquet log through this exact class in timestamp order,
so a feature like "how often has this user been reverted" only counts reverts
that had *already happened* when the edit was made. Serving feeds it the live
stream. One code path means no train/serve skew.

Every feature is computed over the same trailing WINDOW. Unbounded counters
("all edits we've ever seen by this user") would mean something different in
a training row near the start of the log than in a server that has been up for
a week. Training rows start only after a full WINDOW of history (dataset.py),
and the server warms up on exactly one WINDOW.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict, deque

from .events import is_revert_comment, reverted_user_from_comment
from .editwar import EditWarDetector

WINDOW = 12 * 3600.0


class _LRU(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.move_to_end(k)
        if len(self) > self.maxsize:
            self.popitem(last=False)


class OnlineState:
    def __init__(self, max_revs: int = 1_000_000):
        self.user_edits: dict[str, deque[float]] = defaultdict(deque)
        self.user_reverted: dict[str, deque[float]] = defaultdict(deque)
        self.page_edits: dict[str, deque[float]] = defaultdict(deque)
        self.page_reverts: dict[str, deque[float]] = defaultdict(deque)
        self.last_editor: dict[str, tuple[str, float]] = {}
        self.rev_owner: _LRU = _LRU(max_revs)  # rev_id -> (user, title, ts)
        self.editwar = EditWarDetector()
        self.now = 0.0
        self._ops = 0

    @staticmethod
    def _count(d: dict[str, deque[float]], key: str, now: float, window: float = WINDOW) -> int:
        q = d.get(key)
        if not q:
            return 0
        while q and q[0] < now - WINDOW:  # only ever expire the full window
            q.popleft()
        if window >= WINDOW:
            return len(q)
        n = 0  # shorter sub-window: count from the newest end, don't mutate
        for t in reversed(q):
            if t < now - window:
                break
            n += 1
        return n

    def _gc(self) -> None:
        """Drop keys whose windows have fully expired, so memory stays bounded."""
        cutoff = self.now - WINDOW
        for d in (self.user_edits, self.user_reverted, self.page_edits, self.page_reverts):
            for k in [k for k, q in d.items() if not q or q[-1] < cutoff]:
                del d[k]
        self.last_editor = {k: v for k, v in self.last_editor.items() if v[1] >= cutoff}

    def snapshot(self, e: dict) -> dict:
        """Features available at the moment edit `e` arrives (before observing it)."""
        ts, user, title = float(e["timestamp"]), e["user"], e["title"]
        n_edits = self._count(self.user_edits, user, ts)
        n_rev = self._count(self.user_reverted, user, ts)
        last = self.last_editor.get(title)
        if last and last[1] < ts - WINDOW:
            last = None
        return {
            "user_edits_12h": n_edits,
            "user_reverted_12h": n_rev,
            "user_revert_rate": (n_rev + 0.1) / (n_edits + 1.0),
            "page_edits_1h": self._count(self.page_edits, title, ts, 3600),
            "page_edits_12h": self._count(self.page_edits, title, ts),
            "page_reverts_12h": self._count(self.page_reverts, title, ts),
            "secs_since_page_edit": (ts - last[1]) if last else WINDOW,
            "same_user_as_last": float(bool(last) and last[0] == user),
        }

    def observe_edit(self, e: dict) -> dict | None:
        """Update state with an edit. Returns an edit-war alert if one fires."""
        ts, user, title = float(e["timestamp"]), e["user"], e["title"]
        self.now = max(self.now, ts)
        self._ops += 1
        if self._ops % 50_000 == 0:
            self._gc()
        self.user_edits[user].append(ts)
        self.page_edits[title].append(ts)
        prev = self.last_editor.get(title)
        self.last_editor[title] = (user, ts)
        self.rev_owner[int(e["revision_new"])] = (user, title, ts)
        # Fast path for edit wars: revert summaries name the reverted user, so we
        # don't have to wait for the tags-change event.
        if not e["bot"] and is_revert_comment(e.get("comment", "")):
            victim = reverted_user_from_comment(e["comment"]) or (prev[0] if prev else None)
            if victim and victim != user:
                return self.editwar.observe(title, user, victim, ts)
        return None

    def observe_revert(self, r: dict) -> tuple[str, str | None] | None:
        """Update state with an `mw-reverted` event. Returns (reverted_user, reverter)."""
        ts = float(r["revert_ts"])
        self.now = max(self.now, ts)
        owner = self.rev_owner.get(int(r["rev_id"]))
        if owner is None:
            return None
        user, title, _ = owner
        reverter = self.last_editor.get(title, (None, 0))[0]
        if reverter == user:  # self-revert, not a signal
            return user, reverter
        self.user_reverted[user].append(ts)
        self.page_reverts[title].append(ts)
        return user, reverter
