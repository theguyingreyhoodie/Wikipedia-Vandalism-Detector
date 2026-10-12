# wiki-vandalism-detector

Real-time vandalism and edit-war detection on English Wikipedia. Edits come from
Wikimedia EventStreams, which needs no API key. Labels are built from what the
community does later: **an edit that someone else reverts within `H` hours is a
positive**.

```
EventStreams (SSE)
 ├─ mediawiki.recentchange ──────────► edits  ─┐
 └─ mediawiki.revision-tags-change ──► reverts │  (fires when `mw-reverted` is added
         (delayed labels)                      │   to an existing revision)
                                               ▼
                         OnlineState: point-in-time features ◄── same code offline & online
                                               │
              diff (action=compare, cached) ──►│──► MiniLM embedding (cached)
                                               ▼
                    HistGradientBoosting (class-weighted) ──► score, alert, latency log
                                               │
                     monitor: PSI drift · delayed PR-AUC · latency p50/p95/p99
                     retrain: champion/challenger on newest matured window
```

## Quickstart

```bash
python3.14 -m venv .venv && .venv/bin/pip install -e '.[embed,dev]'
# User-Agent contact (defaults to this repo's URL). Without one, Wikimedia treats
# you as "unidentified": 10 API req/min instead of 200.
export WIKIVAND_CONTACT="https://github.com/you/your-fork"

# 1. Collect data. --since replays up to ~7 days of history, so you don't have to wait.
python -m wikivand ingest --since 2026-10-10T00:00:00Z --no-diffs
# Diffs for history: every matured positive + 3 random negatives each (re-weighted later)
python -m wikivand backfill-diffs --days 2 --neg-per-pos 3

# 2. Look at your labels before training on them
python -m wikivand labels

# 3. Train (time-ordered 60/20/20 split, threshold picked on val, reported on test)
python -m wikivand train --days 2 --ablation   # needs > 12h warm-up + 6h horizon of data

# 4. Score live edits; prints 🚩 alerts, ⚔️ edit wars, and latency percentiles every minute
python -m wikivand serve

# 5. Monitor and retrain
python -m wikivand monitor --hours 24
python -m wikivand retrain --days 2             # promotes only if it beats the champion
python -m wikivand editwars --days 2            # offline replay of the edit-war detector
```

`serve` is a superset of `ingest`: it keeps writing the raw logs, so labels for
today's predictions arrive on their own.

## The hard parts

### Labels arrive late
A revert can happen seconds or days after the edit. Each edit gets a `label_status`
([labels.py](wikivand/labels.py)):

| status | meaning |
|---|---|
| `pos` | reverted by *someone else* within `H` (default 6h, `WIKIVAND_HORIZON_H`) |
| `neg` | `H` has fully passed, measured against **data coverage** rather than wall-clock time, with no revert inside it |
| `pending` | too new to know. Excluded from training *and* evaluation |
| `self_revert` | the editor undid their own edit. Dropped |

Why `pending` matters: positives become known early but negatives only become
known after `H`. If you keep early positives and drop unknown negatives, the
newest slice of data looks far more vandalised than it is. `python -m wikivand labels`
prints the time-to-revert distribution and the share of reverts that land inside
`H`, so you can choose the horizon from data rather than by guessing.

The tags-change event doesn't say who reverted. The reverter is inferred as the
latest editor of that page at the moment the tag was applied
(`attach_reverter`), which is also how self-reverts are detected.

### Leakage and train/serve skew
* **Point-in-time features.** "How often has this user been reverted" must only
  count reverts that had already happened when the edit was made. Training
  replays the parquet log in timestamp order through `OnlineState`
  ([state.py](wikivand/state.py)), which is the same class that serves live
  traffic. One code path removes that whole class of skew.
* **Same history depth on both sides.** State features are trailing 12h
  windows. Training rows start 12h after the log begins, and the server warms up
  on exactly 12h. The first version used lifetime counters, and `monitor` flagged
  `user_prior_edits` drift on the first live run: early training rows had no
  history, while the server had a full day of it.
* **Shape guard.** `model.matrix` refuses to build a matrix whose width doesn't
  match the feature list. The raw `namespace` column once collided with the
  `namespace` feature, so training saw 38 columns and serving saw 37.
* **Diffs are fetched at ingest time.** Vandalism is often revision-deleted
  later, so a backfill sees "missing content" mostly on positives. The diff cache
  records a `status` per revision so you can measure this. Prefer data that was
  ingested live.
* **Chronological splits only.** No shuffling.
* The model artifact stores the embedder name, and `serve` refuses to run with a
  different one.

### Class imbalance
About 4–5% of article edits by non-bots get reverted. The pipeline uses
`class_weight="balanced"` and reports **PR-AUC**, recall at precision
50/80/90%, precision in the top 1% of scores, and alerts per 1k edits. It does
not report accuracy, which is meaningless at this base rate. The alert
threshold is the lowest score that reaches `--target-precision` (default 0.5,
suited to a human review queue) on the validation split.

Diffs are expensive (one API call each, 200/min), so history is
**label-stratified**: every positive plus K negatives. `dataset.py` attaches
importance weights (class size ÷ rows kept), and every metric and threshold is
computed with them. Reported precision and base rate therefore describe the real
traffic, not the 25%-positive sample. A test checks that weighted metrics on a
down-sample match the full population.

### Concept drift
Vandal behaviour, editing patterns (time of day, news events) and even the
platform change; for example, IP editors are now temporary accounts named
`~2026-…`. `monitor` reports:
* **label-free, available now:** PSI of every feature and of the score
  distribution against the training reference. Cyclic features (`hour_utc`)
  are skipped, because an hour of traffic always "drifts" from a full-day
  reference
* **label-based, available after `H`:** PR-AUC on predictions whose labels have
  matured, bucketed over time
* a `retrain_recommended` list of reasons

`retrain` trains a challenger on the newest matured window and scores the
champion on the *same* held-out rows. It promotes the challenger only if
PR-AUC improves by `--min-gain`.

### Latency
Every scored edit logs per-stage timings: `queue`, `diff` (the Wikipedia API
round trip), `features`, `embed`, `predict` and the total. It also logs
`event_lag_ms`, the time from the edit being saved to us receiving it. `serve`
prints rolling p50/p95/p99, and `monitor` reports all of them. See the results
below: the diff fetch is about 97% of the total.

### Edit wars
[editwar.py](wikivand/editwar.py) keeps revert edges (reverter → reverted user)
per page over 24h. Edges are parsed from revert summaries ("Undid revision N by
[[Special:Contributions/X|X]]", rollback, Twinkle), so detection is immediate.
It fires on **mutual reverting** (A↔B with ≥4 reverts) or a **3RR breach**
(more than 3 reverts by one user). Alerts are written to
`data/alerts/editwars.jsonl`.

## Features
* **metadata:** anon/temp account, size delta, blanking, minor, edit summary
  shape (empty, section-only, tool, revert), digits in the username, hour
* **state (point-in-time):** user's prior edits and reverts, page activity and
  reverts in 1h/24h, seconds since the last page edit, same editor as last
* **diff text:** added/removed size, uppercase/punctuation ratios, character
  runs ("!!!!!!"), profanity lexicon hits (added *and* removed), pronouns,
  links added, refs removed, markup ratio
* **embedding:** `all-MiniLM-L6-v2` on added text plus a slice of removed text,
  compressed with PCA to 24 dims inside the pipeline. `--embedder hashing` gives
  a dependency-free fallback. `train --ablation` shows what each family adds.

## Data layout
```
data/raw/date=YYYY-MM-DD/edits_*.parquet        recentchange edits (append-only)
data/reverts/date=YYYY-MM-DD/reverts_*.parquet  mw-reverted events
data/predictions/date=…/predictions_*.parquet   score, latency per stage, feature values
data/cache.sqlite                               diffs + embeddings
data/models/<version>/{model.joblib,meta.json}  + data/models/CHAMPION
```

## Results (first run, 2026-10-11)

These numbers come from about 34h of enwiki history replayed from EventStreams. With
12h of feature warm-up and a 6h label horizon, that leaves ~16h of labelled
edits: 6,448 article edits by non-bots with diffs, stratified to 2,191 positives.
The split is chronological 60/20/20, and every metric is importance-weighted
back to the real ~4% base rate.

| features (test split) | PR-AUC |
|---|---|
| metadata only | 0.140 |
| + point-in-time state | 0.240 |
| + diff text stats | 0.229 |
| + MiniLM embedding | **0.265** |

Final model on test: PR-AUC 0.265 at a 4.1% base rate (6.5× lift), ROC-AUC
0.83, precision 54% in the top 1% of scores. The operating point chosen for 50%
precision on val delivered 37% precision at 16% recall on test. The threshold
didn't fully transfer across a few hours, which is what the delayed
evaluation in `monitor` is for. With only ~600 test positives, gaps of ±0.02
between ablation rows are noise. Collect more days before trusting the small
differences.

**Live latency** (`serve`, ~450 edits scored in 5 min, laptop CPU):

| stage | p50 | p95 |
|---|---|---|
| diff fetch (Wikipedia API) | 667 ms | 1.4 s |
| features | 0.1 ms | 1.4 ms |
| MiniLM embed | 12 ms | 27 ms |
| predict | 14 ms | 44 ms |
| **total** | **~700 ms** | **~1.4–1.7 s** |
| EventStreams lag (edit saved → received) | ~3.9 s | ~10 s |

**What monitoring caught on the first live runs.** The drift report turned up
four real bugs before any labels matured:
1. Lifetime user counters gave train/serve skew. Fixed with trailing 12h
   windows and matched warm-up.
2. A 1h counter was truncating the shared deque behind the 12h counter.
3. The drift reference was the positive-enriched sample, not resampled by weight.
4. A QuickCategories batch (2,553 edits in 52 min from one non-bot account)
   dominated an hour of training data. Batch-tool edits are now ineligible,
   like bots.

After those fixes, score PSI live vs training was 0.06 with no feature shifted.
An offline replay of the live-scored edits reproduces their state features
exactly (98–100%), which checks train/serve parity directly.

## EventStreams and API gotchas (learned the hard way)
* **Don't multiplex streams on one connection for replay.** `recentchange,revision-tags-change`
  on one connection with `?since=` was ~50× slower than separate connections.
  `sse.iter_merged` opens one per stream.
* **`httpx.iter_lines` is CPU-bound at firehose rates.** Parsing raw byte chunks,
  with a byte-level `"enwiki"` prefilter before `json.loads`, gave a 4× speedup.
* **The two streams replay at different speeds.** Label maturity is measured
  against `min(latest edit time, latest revert time)` in event time, not
  ingestion time. Otherwise a replay marks edits as clean negatives before their
  reverts arrive.
* **`revision-create` no longer carries revert info.** Labels come from
  `mw-reverted` being added in `revision-tags-change`.
* **Rate limits are by identity.** Send a User-Agent with contact info, pace
  requests process-wide, and honour `Retry-After` (`diffs.RateLimiter`).
* **SIGTERM flushes buffers**, so `kill` doesn't drop the last minute of events.

## Known limitations
* "Reverted" ≠ "vandalism". Good-faith mistakes, content disputes and
  edit-war collateral get reverted too. Treat the score as "likely to be
  reverted".
* Gaps in ingestion silently turn positives into negatives, because a missed
  revert looks like no revert. Keep the ingester running, or replay the gap with
  `--since` (EventStreams keeps ~7 days).
* Revert edges for edit wars come only from edit summaries; manual reverts with
  custom summaries are missed.
* Historical diffs are fetched up to ~30h after the edit. Only 31 of ~12k were
  already revision-deleted (97% of those positive). They are dropped and
  reported, but this is still a small selection bias. Diffs fetched live by
  `serve` don't have it.
* Next feature to add: `performer.user_edit_count` / `user_registration_dt`
  from the `revision-create` stream. They are lifetime and point-in-time correct
  by construction, so the 12h window doesn't limit them.
