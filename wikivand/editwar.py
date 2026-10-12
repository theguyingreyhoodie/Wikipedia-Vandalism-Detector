"""Edit-war detection from revert edges (reverter -> reverted user) per page.

Fires when, within a sliding window on one page, either
  * two users have reverted each other and there are >= `min_reverts` reverts, or
  * one user has made more than `rrr` reverts (Wikipedia's three-revert rule)
    *and* was reverted on that page too. Without that second condition, every
    vandal patroller cleaning up after several different vandals would trip 3RR
    (reverting obvious vandalism is exempt on Wikipedia as well).
Bot reverts are ignored by the caller.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque


class EditWarDetector:
    def __init__(self, window_s: float = 86400, min_reverts: int = 4, rrr: int = 3,
                 cooldown_s: float = 3600):
        self.window_s = window_s
        self.min_reverts = min_reverts
        self.rrr = rrr
        self.cooldown_s = cooldown_s
        self.edges: dict[str, deque[tuple[float, str, str]]] = defaultdict(deque)
        self.last_alert: dict[str, float] = {}

    def observe(self, title: str, reverter: str, reverted: str, ts: float) -> dict | None:
        q = self.edges[title]
        q.append((ts, reverter, reverted))
        while q and q[0][0] < ts - self.window_s:
            q.popleft()
        if len(q) < 2 or ts - self.last_alert.get(title, -1e18) < self.cooldown_s:
            return None

        pairs = {(a, b) for _, a, b in q}
        mutual = sorted({tuple(sorted(p)) for p in pairs if (p[1], p[0]) in pairs})
        by_user = Counter(a for _, a, _ in q)
        was_reverted = {b for _, _, b in q}
        rrr_violators = sorted(u for u, n in by_user.items() if n > self.rrr and u in was_reverted)

        if (mutual and len(q) >= self.min_reverts) or rrr_violators:
            self.last_alert[title] = ts
            return {
                "title": title,
                "ts": ts,
                "reverts_in_window": len(q),
                "mutual_pairs": [list(p) for p in mutual],
                "rrr_violators": rrr_violators,
                "reverts_by_user": dict(by_user),
            }
        return None
