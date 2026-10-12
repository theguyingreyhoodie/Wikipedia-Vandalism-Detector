import numpy as np
import pandas as pd

from wikivand.diffs import parse_diff_html, wiki_domain
from wikivand.editwar import EditWarDetector
from wikivand.events import edit_row, is_anonymous, reverted_user_from_comment, revert_row
from wikivand.labels import build_labels
from wikivand.monitor import psi
from wikivand.sse import parse_sse
from wikivand.state import OnlineState

H = 6 * 3600


def edit(rev, ts, user="Alice", title="Page", comment="", old=None):
    return {"event_id": f"enwiki_{rev}", "timestamp": ts, "user": user, "bot": False, "minor": False,
            "title": title, "namespace": 0, "comment": comment, "revision_new": rev,
            "revision_old": old if old is not None else rev - 1, "length_new": 100,
            "length_old": 90, "wiki": "enwiki", "ingestion_ts": ts}


def revert(rev, rev_ts, revert_ts, title="Page"):
    return {"wiki": "enwiki", "rev_id": rev, "title": title, "namespace": 0, "rev_ts": rev_ts,
            "revert_ts": revert_ts, "ingestion_ts": revert_ts}


def test_parse_sse():
    lines = ["event: message", "id: [1]", 'data: {"a": 1}', "", ":ok", "data: {bad", "", 'data: {"b": 2}', ""]
    assert list(parse_sse(lines)) == [("[1]", {"a": 1}), (None, {"b": 2})]
    # prefilter skips the payload but still reports the id for resumption
    assert list(parse_sse(lines, contains=b'"b"')) == [("[1]", {}), (None, {"b": 2})]


def test_parse_diff_html():
    body = (
        '<td class="diff-deletedline diff-side-deleted"><div>old <del class="diffchange">bad</del> x</div></td>'
        '<td class="diff-addedline diff-side-added"><div>old <ins class="diffchange">HAHA</ins> x</div></td>'
        '<td class="diff-addedline diff-side-added"><div>whole &amp; new line</div></td>'
    )
    added, removed = parse_diff_html(body)
    assert added == "HAHA\nwhole & new line"
    assert removed == "bad"


def test_events():
    assert is_anonymous("~2026-54444-90") and is_anonymous("1.2.3.4") and is_anonymous("2001:db8::1")
    assert not is_anonymous("Doomsdayer520")
    assert reverted_user_from_comment("Undid revision 123 by [[Special:Contributions/Bob|Bob]] (talk)") == "Bob"
    assert wiki_domain("enwiki") == "en.wikipedia.org"
    from wikivand.events import is_batch_tool
    assert is_batch_tool("+[[Category:X]] ([[toolforge:quickcategories/batch/18648/|QuickCategories batch #18648]])")
    assert is_batch_tool("Typo fixing, [[WP:AWB/T|typo(s) fixed]]: a → b using [[Project:AWB|AWB]]")
    assert not is_batch_tool("/* Career */ still working")
    ev = {"type": "edit", "wiki": "enwiki", "timestamp": 10, "user": "A", "title": "T", "namespace": 0,
          "revision": {"new": 5, "old": 4}, "length": {"new": 3, "old": 1}}
    assert edit_row(ev)["event_id"] == "enwiki_5"
    tag = {"database": "enwiki", "rev_id": 5, "page_title": "Some_Page", "page_namespace": 0,
           "rev_timestamp": "2026-10-11T14:55:31Z", "meta": {"dt": "2026-10-11T15:03:22.475Z"},
           "tags": ["visualeditor", "mw-reverted"], "prior_state": {"tags": ["visualeditor"]}}
    r = revert_row(tag)
    assert r["title"] == "Some Page" and r["revert_ts"] - r["rev_ts"] > 400
    tag["prior_state"]["tags"].append("mw-reverted")
    assert revert_row(tag) is None  # tag already present: not a new revert


def test_labels_statuses():
    edits = pd.DataFrame([
        edit(1, 0, "Vandal"), edit(2, 60, "Fixer"),                  # 1 reverted by Fixer at 70s
        edit(3, 1000, "Oops", title="Q"), edit(4, 1010, "Oops", title="Q"),  # 3 self-reverted
        edit(5, 2000, "Slow", title="R"), edit(6, 2000 + H + 100, "X", title="R"),  # 5 reverted late
        edit(7, 3000, "Clean", title="S"),
        edit(8, 30 * 3600, "Recent", title="T"),                     # too new to know
    ])
    reverts = pd.DataFrame([revert(1, 0, 70), revert(3, 1000, 1012, "Q"), revert(5, 2000, 2000 + H + 105, "R")])
    lab = build_labels(edits, reverts, H, coverage_end=31 * 3600).set_index("revision_new")
    assert lab.loc[1, "label_status"] == "pos" and lab.loc[1, "reverter"] == "Fixer"
    assert lab.loc[3, "label_status"] == "self_revert"
    assert lab.loc[5, "label_status"] == "neg"
    assert lab.loc[7, "label_status"] == "neg"
    assert lab.loc[8, "label_status"] == "pending" and np.isnan(lab.loc[8, "label"])


def test_state_is_point_in_time():
    s = OnlineState()
    s.observe_edit(edit(1, 0, "V"))
    snap_before = s.snapshot(edit(3, 50, "V"))
    s.observe_edit(edit(2, 60, "Fixer"))
    s.observe_revert(revert(1, 0, 61))
    snap_after = s.snapshot(edit(3, 100, "V"))
    assert snap_before["user_reverted_12h"] == 0
    assert snap_after["user_reverted_12h"] == 1 and snap_after["page_reverts_12h"] == 1
    # outside the window, history no longer counts
    later = s.snapshot(edit(4, 13 * 3600, "V"))
    assert later["user_reverted_12h"] == 0 and later["user_edits_12h"] == 0


def test_self_revert_not_counted():
    s = OnlineState()
    s.observe_edit(edit(1, 0, "V"))
    s.observe_edit(edit(2, 10, "V"))
    s.observe_revert(revert(1, 0, 11))
    assert len(s.user_reverted["V"]) == 0


def test_editwar_mutual_and_3rr():
    d = EditWarDetector(min_reverts=4)
    assert d.observe("P", "A", "B", 0) is None
    assert d.observe("P", "B", "A", 10) is None  # mutual but only 2 reverts
    d.observe("P", "A", "B", 20)
    alert = d.observe("P", "B", "A", 30)
    assert alert and alert["mutual_pairs"] == [["A", "B"]]
    # patroller reverting four different vandals: not a war
    d2 = EditWarDetector()
    assert not any(d2.observe("Q", "C", f"u{i}", i) for i in range(4))
    # 4 reverts by C while being reverted by D: 3RR
    d3 = EditWarDetector(min_reverts=99)
    d3.observe("R", "D", "C", 0)
    alerts = [d3.observe("R", "C", f"u{i}", i + 1) for i in range(4)]
    assert alerts[-1] and alerts[-1]["rrr_violators"] == ["C"]


def test_editwar_via_state_comments():
    s = OnlineState()
    s.observe_edit(edit(1, 0, "A"))
    alerts = []
    for i, (u, v) in enumerate([("B", "A"), ("A", "B"), ("B", "A"), ("A", "B")]):
        c = f"Undid revision {i} by [[Special:Contributions/{v}|{v}]]"
        alerts.append(s.observe_edit(edit(10 + i, 100 + i, u, comment=c)))
    assert alerts[-1] is not None


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.05
    assert psi(a, rng.normal(1.5, size=5000)) > 0.25


def test_train_on_synthetic():
    from wikivand.features import FEATURES
    from wikivand.model import score, train

    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df["timestamp"] = np.arange(n)
    df["label"] = (df.bad_words + df.is_anon + rng.normal(scale=0.5, size=n) > 2.2).astype(float)
    emb = rng.normal(size=(n, 16)).astype(np.float32)
    art, rep = train(df, emb)
    assert rep["test"]["pr_auc"] > 0.5
    assert score(art, df.iloc[:5], emb[:5]).shape == (5,)


def test_weighted_metrics_match_full_population():
    """Down-sampling negatives + importance weights ~= metrics on the full data."""
    from wikivand.model import metrics

    rng = np.random.default_rng(1)
    n = 40000
    y = (rng.random(n) < 0.03).astype(int)
    s = rng.normal(size=n) + 1.5 * y
    full = metrics(y, s, threshold=2.0)
    keep = (y == 1) | (rng.random(n) < 0.1)
    w = np.where(y[keep] == 1, 1.0, 10.0)
    sub = metrics(y[keep], s[keep], threshold=2.0, w=w)
    for k in ("base_rate", "pr_auc", "precision", "alerts_per_1k_edits"):
        assert abs(sub[k] - full[k]) / full[k] < 0.1, k


def test_serving_matrix_matches_training(tmp_path):
    """Regression: raw `namespace` + feature `namespace` once produced 38 columns in training."""
    from wikivand.features import FEATURES, featurize
    from wikivand.model import matrix
    from wikivand.state import OnlineState

    e = edit(1, 0)
    feats = featurize(e, OnlineState().snapshot(e), {"status": "ok", "added": "x", "removed": ""})
    raw = pd.DataFrame([e])
    train_like = pd.concat([raw.drop(columns=[c for c in FEATURES if c in raw]),
                            pd.DataFrame([feats])], axis=1)
    emb = np.zeros((1, 4), np.float32)
    assert matrix(train_like, emb, FEATURES).shape == matrix(pd.DataFrame([feats]), emb, FEATURES).shape


def test_short_window_does_not_truncate_long_window():
    s = OnlineState()
    for i, t in enumerate([0, 2 * 3600, 5 * 3600]):
        s.observe_edit(edit(i + 1, t, f"U{i}"))
    snap = s.snapshot(edit(9, 5 * 3600 + 60, "X"))
    assert snap["page_edits_1h"] == 1
    assert snap["page_edits_12h"] == 3
    assert s.snapshot(edit(10, 5 * 3600 + 61, "X"))["page_edits_12h"] == 3  # still 3 after a 1h count
