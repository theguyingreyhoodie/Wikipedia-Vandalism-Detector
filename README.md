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
# Faster: replay in parallel segments, each stopping at --until
# python -m wikivand ingest --no-diffs --since 2026-10-05T09:00:00Z --until 2026-10-07T03:00:00Z
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
# --end trains on an older window, e.g. to test whether retraining pays off
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
  on exactly 12h. Lifetime counters would give training rows near the start of
  the log no history, while a long-running server has days of it.
* **Shape guard.** `model.matrix` refuses to build a matrix whose width doesn't
  match the feature list. For example, the raw `namespace` column has the same
  name as the `namespace` feature and would otherwise be counted twice.
* **Diffs are fetched at ingest time.** Vandalism is often revision-deleted
  later, so a backfill sees "missing content" mostly on positives. The diff cache
  records a `status` per revision so you can measure this. Prefer data that was
  ingested live.
* **Chronological splits only.** No shuffling.
* The model artifact stores the embedder name, and `serve` refuses to run with a
  different one.

### Class imbalance
Only a few percent of article edits by non-bots get reverted. The pipeline uses
`class_weight="balanced"` and reports **PR-AUC**, recall at precision
50/80/90%, precision in the top 1% of scores, and alerts per 1k edits. It does
not report accuracy, which is meaningless at this base rate. The alert
threshold is the lowest score that reaches `--target-precision` (default 0.5,
suited to a human review queue) on the validation split.

Diffs are expensive (one API call each, 200/min), so history is
**label-stratified**: every positive plus K negatives. `dataset.py` attaches
importance weights (class size ÷ rows kept), and every metric and threshold is
computed with them. Reported precision and base rate therefore describe the real
traffic, not the positive-enriched sample. A test checks that weighted metrics on a
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
prints rolling p50/p95/p99, and `monitor` reports all of them. The diff fetch,
a network round trip to Wikipedia, dominates the total.

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

## Results (7-day run, 2026-10-05 → 2026-10-12)

**Data.** Seven days of English Wikipedia, the most EventStreams keeps: 1.17M
edits and 139k revert events, replayed in four parallel segments with no gaps
longer than 5 minutes. Of ~620k matured article edits by non-bot accounts,
**4.5% were reverted by someone else within 6h**. Diffs were fetched for a random
42k of {every reverted edit + 2 normal edits each}. That gives 38k training rows
after the 12h warm-up and the drops below. Every metric is weighted back to the
real base rate.

**How labels arrive.** The median revert lands 18.5 minutes after the edit, but
the 90th percentile takes 18h. The 6h horizon captures 79% of reverts. A longer
horizon catches more slow reverts, but every label then arrives later.

**Feature ablation.** The test split is the newest ~day (7,600 edits, 4.3%
reverted), with the same split and model for each row:

| features | PR-AUC | ROC-AUC |
|---|---|---|
| edit metadata | 0.229 | 0.814 |
| + point-in-time user/page state | 0.347 | 0.872 |
| + diff text stats | **0.390** | 0.879 |
| + MiniLM embedding | 0.388 | 0.879 |

State features and diff text both add clear lift. On top of the hand-built diff
features, the MiniLM embedding adds nothing measurable at this data size.

**Operating point.** The threshold picked for 50% precision on validation gives
**58.5% precision at 28% recall** on test, about 21 alerts per 1,000 edits.
Precision in the top 1% of scores is 72%. PR-AUC is 0.384 on val and 0.388 on
test, so the threshold transfers.

**Retraining (champion/challenger).** A champion trained only on data up to
Oct 10 12:00 UTC was compared with a challenger trained on everything up to
Oct 12 03:00, both scored on the same newest held-out day:

| model | PR-AUC | ROC-AUC | precision / recall at its threshold |
|---|---|---|---|
| champion (data to Oct 10) | 0.388 | 0.877 | 53.5% / 30.9% |
| challenger (data to Oct 12) | 0.388 | 0.879 | 58.5% / 28.1% |

Two extra days bought nothing measurable, so there was no meaningful drift that
week. The promotion gate (minimum +0.005 PR-AUC) correctly kept the champion.

**Live serving.** `serve` scored 637 edits in 10 minutes, laptop CPU:

| stage | p50 | p95 | p99 |
|---|---|---|---|
| diff fetch (Wikipedia API) | 659 ms | 1.65 s | 2.34 s |
| features | 0.2 ms | 1.5 ms | 3.4 ms |
| MiniLM embed | 12.8 ms | 26.9 ms | 28.3 ms |
| predict | 12.6 ms | 36.3 ms | 39.0 ms |
| **total** | **676 ms** | **1.66 s** | **2.38 s** |
| EventStreams lag (edit saved → received) | 4.1 s | 11.6 s | 20.6 s |

The diff round trip is 97% of the time; the model side is about 25 ms. Live
score PSI against the training reference was 0.037, with no feature above
0.25, so there was no drift flag. The live alert rate was 26.7 per 1,000 edits.

**Edit wars.** Replaying the week through the detector gave 57 alerts.

**Data caveats.** 148 diffs were already revision-deleted when fetched (89% of
them reverted) and 70 revisions no longer existed. Both groups were dropped.
Another 255 fetches failed during a brief network outage and were skipped,
which is unrelated to labels.

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
* Historical diffs are fetched hours or days after the edit. Diffs that were
  revision-deleted by then (nearly always vandalism) are dropped and reported,
  but this is still a small selection bias. Diffs fetched live by `serve` don't
  have it.
* Next feature to add: `performer.user_edit_count` / `user_registration_dt`
  from the `revision-create` stream. They are lifetime and point-in-time correct
  by construction, so the 12h window doesn't limit them.
