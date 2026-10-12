"""Central configuration. Everything is overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"WIKIVAND_{name}", default)


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(_env("DATA", "data"))
    wiki: str = _env("WIKI", "enwiki")
    # Namespaces we train on / score. Edits in other namespaces are still ingested.
    namespaces: tuple[int, ...] = tuple(int(n) for n in _env("NAMESPACES", "0").split(","))
    # An edit reverted within this many hours is a positive. An edit is only
    # labelled (positive OR negative) once this much time has passed.
    label_horizon_h: float = float(_env("HORIZON_H", "6"))
    # Wikimedia rate-limits by identity: a User-Agent with contact info (email or
    # URL) gets 200 req/min; without one you're "unidentified" at 10 req/min.
    contact: str = _env("CONTACT", "https://github.com/theguyingreyhoodie/Wikipedia-Vandalism-Detector")
    stream_url: str = "https://stream.wikimedia.org/v2/stream/"
    # Wikipedia API requests/second for diff fetching (shared by all worker threads).
    api_rps_override: str = _env("API_RPS", "")
    embed_model: str = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def user_agent(self) -> str:
        contact = f" ({self.contact})" if self.contact else ""
        return f"wiki-vandalism-detector/0.1{contact} python-httpx"

    @property
    def api_rps(self) -> float:
        if self.api_rps_override:
            return float(self.api_rps_override)
        return 3.0 if self.contact else 10 / 60  # stay just under the published limits

    @property
    def horizon_s(self) -> float:
        return self.label_horizon_h * 3600

    @property
    def raw_dir(self) -> Path:  # recentchange edits
        return self.data_dir / "raw"

    @property
    def reverts_dir(self) -> Path:  # mw-reverted tag events
        return self.data_dir / "reverts"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def cache_db(self) -> Path:  # diff + embedding cache
        return self.data_dir / "cache.sqlite"


CONFIG = Config()
