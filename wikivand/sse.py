"""Minimal, reconnecting Server-Sent-Events client for Wikimedia EventStreams."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Iterable, Iterator

import httpx

log = logging.getLogger(__name__)


def parse_sse(lines: Iterable[str | bytes], contains: bytes | None = None
              ) -> Iterator[tuple[str | None, dict]]:
    """Turn raw SSE lines into (event_id, json_payload) pairs.

    `contains` is a cheap byte-level prefilter: events whose data doesn't contain
    it are skipped without JSON parsing (most of the firehose is other wikis).
    The event id is still reported so resumption stays exact.
    """
    data: list[bytes] = []
    event_id: str | None = None
    for line in lines:
        if isinstance(line, str):
            line = line.encode()
        line = line.rstrip(b"\r")
        if not line:
            if data:
                raw = b"\n".join(data)
                if contains is None or contains in raw:
                    try:
                        yield event_id, json.loads(raw)
                    except json.JSONDecodeError:
                        log.debug("skipping malformed event")
                elif event_id:
                    yield event_id, {}
            data, event_id = [], None
        elif line.startswith(b"data:"):
            data.append(line[5:].lstrip())
        elif line.startswith(b"id:"):
            event_id = line[3:].strip().decode()
        # "event:" and ":" comment lines are ignored


def _iter_byte_lines(chunks: Iterable[bytes]) -> Iterator[bytes]:
    buf = b""
    for chunk in chunks:
        buf += chunk
        *lines, buf = buf.split(b"\n")
        yield from lines


def iter_events(
    streams: list[str],
    *,
    base_url: str,
    user_agent: str,
    since: str | None = None,
    contains: bytes | None = None,
) -> Iterator[dict]:
    """Yield events forever, resuming via Last-Event-ID after disconnects.

    `since` (ISO-8601 or epoch ms) replays history; EventStreams keeps ~7 days.
    """
    url = base_url + ",".join(streams)
    last_id: str | None = None
    backoff = 1.0
    while True:
        headers = {"User-Agent": user_agent, "Accept": "text/event-stream"}
        params = {}
        if last_id:
            headers["Last-Event-ID"] = last_id
        elif since:
            params["since"] = since
        try:
            timeout = httpx.Timeout(10.0, read=60.0)
            with httpx.stream("GET", url, params=params, headers=headers, timeout=timeout) as r:
                r.raise_for_status()
                backoff = 1.0
                lines = _iter_byte_lines(r.iter_raw(chunk_size=1 << 16))
                for event_id, payload in parse_sse(lines, contains):
                    if event_id:
                        last_id = event_id
                    if payload:
                        yield payload
        except (httpx.HTTPError, httpx.StreamError) as e:
            log.warning("stream disconnected (%s); reconnecting in %.0fs", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


def iter_merged(streams: list[str], *, base_url: str, user_agent: str, since: str | None = None,
                contains: bytes | None = None, maxsize: int = 20000) -> Iterator[dict]:
    """One connection per stream, merged in arrival order.

    EventStreams can multiplex streams on one connection ("a,b"), but historical
    replay through a multiplexed connection is ~50x slower than separate ones.
    """
    q: queue.Queue[dict] = queue.Queue(maxsize=maxsize)

    def pump(stream: str) -> None:
        for ev in iter_events([stream], base_url=base_url, user_agent=user_agent, since=since,
                              contains=contains):
            q.put(ev)

    for s in streams:
        threading.Thread(target=pump, args=(s,), daemon=True, name=f"sse-{s}").start()
    while True:
        yield q.get()
