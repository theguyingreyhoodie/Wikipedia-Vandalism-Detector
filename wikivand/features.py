"""Hand-built features: edit metadata + point-in-time state + diff text."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .events import is_anonymous, is_revert_comment

# Small, deliberately boring lexicon; the embedding model carries the nuance.
_BAD = re.compile(
    r"\b(fuck\w*|shit\w*|bitch\w*|ass(hole)?|dick\w*|cock|penis|vagina|gay|fag\w*|nigg\w*|"
    r"retard\w*|stupid|idiot\w*|dumb|suck[s]?|poop\w*|butt\w*|lol+|lmao|haha\w*|hi+|hello|"
    r"sex\w*|porn\w*|boob\w*|crap|damn|hate|loser|ugly)\b",
    re.I,
)
_PRONOUN = re.compile(r"\b(i|me|my|you|your|u|ur|we)\b", re.I)
_SECTION_ONLY = re.compile(r"^\s*/\*.*\*/\s*$")
_TOOL = re.compile(r"(HotCat|AWB|Twinkle|WP:|\[\[WP|via|bot|script|IABot|Citation bot)", re.I)
_RUN = re.compile(r"(.)\1{4,}")

STATE_FEATURES = [
    "user_edits_12h", "user_reverted_12h", "user_revert_rate", "page_edits_1h",
    "page_edits_12h", "page_reverts_12h", "secs_since_page_edit", "same_user_as_last",
]
META_FEATURES = [
    "is_anon", "is_minor", "namespace", "size_delta_log", "len_old_log", "rel_change",
    "blanking", "comment_len", "comment_empty", "comment_section_only", "comment_tool",
    "comment_revert", "user_digit_ratio", "hour_utc",
]
TEXT_FEATURES = [
    "diff_ok", "added_chars_log", "removed_chars_log", "added_words", "upper_ratio",
    "digit_ratio", "punct_ratio", "longest_run", "bad_words", "bad_words_removed",
    "pronouns", "exclaims", "links_added", "refs_removed", "markup_ratio",
]
FEATURES = META_FEATURES + STATE_FEATURES + TEXT_FEATURES


def _slog(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def meta_features(e: dict) -> dict:
    user, comment = e["user"], e.get("comment") or ""
    lo, ln = int(e["length_old"]), int(e["length_new"])
    d = ln - lo
    return {
        "is_anon": float(is_anonymous(user)),
        "is_minor": float(bool(e["minor"])),
        "namespace": float(e["namespace"]),
        "size_delta_log": _slog(d),
        "len_old_log": math.log1p(lo),
        "rel_change": d / max(lo, 1),
        "blanking": float(lo > 500 and ln < 0.1 * lo),
        "comment_len": float(len(comment)),
        "comment_empty": float(not comment.strip()),
        "comment_section_only": float(bool(_SECTION_ONLY.match(comment))),
        "comment_tool": float(bool(_TOOL.search(comment))),
        "comment_revert": float(is_revert_comment(comment)),
        "user_digit_ratio": sum(c.isdigit() for c in user) / max(len(user), 1),
        "hour_utc": float(datetime.fromtimestamp(int(e["timestamp"]), tz=timezone.utc).hour),
    }


def text_features(diff: dict | None) -> dict:
    ok = diff is not None and diff.get("status") == "ok"
    a = (diff or {}).get("added") or ""
    r = (diff or {}).get("removed") or ""
    letters = [c for c in a if c.isalpha()]
    n = max(len(a), 1)
    return {
        "diff_ok": float(ok),
        "added_chars_log": math.log1p(len(a)),
        "removed_chars_log": math.log1p(len(r)),
        "added_words": float(len(a.split())),
        "upper_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "digit_ratio": sum(c.isdigit() for c in a) / n,
        "punct_ratio": sum(c in "!?.,;:" for c in a) / n,
        "longest_run": float(max((len(m.group(0)) for m in _RUN.finditer(a)), default=0)),
        "bad_words": float(len(_BAD.findall(a))),
        "bad_words_removed": float(len(_BAD.findall(r))),
        "pronouns": float(len(_PRONOUN.findall(a))),
        "exclaims": float(a.count("!")),
        "links_added": float(a.count("http")),
        "refs_removed": float(r.count("<ref") - a.count("<ref")),
        "markup_ratio": sum(c in "[]{}|=<>'" for c in a) / n,
    }


def featurize(e: dict, state_feats: dict, diff: dict | None) -> dict:
    return {**meta_features(e), **state_feats, **text_features(diff)}


def embed_text(diff: dict | None, max_chars: int = 1000) -> str:
    """Text fed to the embedding model: what was added, with a hint of what was removed."""
    if not diff or diff.get("status") != "ok":
        return ""
    a = (diff.get("added") or "")[:max_chars]
    r = (diff.get("removed") or "")[: max_chars // 4]
    return f"added: {a}\nremoved: {r}"
