# EquiRisk

**Equity risk classification and RAG-powered analysis for the Nifty Midcap 150.**

EquiRisk ingests five years of daily market data and recent news for 150 Indian
midcap stocks, processes it through a PySpark ETL pipeline on Amazon S3, engineers
52 features, trains a classifier to predict which stocks will be riskiest over the
coming month, and serves the results through a Streamlit dashboard with a
retrieval-augmented chat assistant.

Built as a CDAC capstone project.

---

## Table of contents

1. [What the project does](#1-what-the-project-does)
2. [Architecture](#2-architecture)
3. [Data model](#3-data-model)
4. [Results](#4-results)
5. [Setup](#5-setup)
6. [Running the pipeline](#6-running-the-pipeline)
7. [The dashboard](#7-the-dashboard)
8. [Repository layout](#8-repository-layout)
9. [Design decisions](#9-design-decisions)
10. [Known limitations](#10-known-limitations)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What the project does

**The question.** Given everything observable about a stock today, will it be among
the riskier third of its peer group over the next 30 trading days?

"Risk" is realised volatility — the standard deviation of daily returns over the
forward window. Each stock is ranked against the other 149 on the same date and
bucketed into Low, Medium or High.

**Why relative rather than absolute.** Asking "will volatility exceed 0.023?"
sounds more natural but is largely unanswerable: whether any given month clears a
fixed threshold depends on market-wide events — crashes, rate decisions, elections —
that daily technicals cannot forecast. Relative volatility rank, by contrast, is a
persistent *stock characteristic*. A name that is volatile relative to its peers in
calm markets is usually volatile relative to them in turbulent ones. See
[Design decisions](#9-design-decisions).

**What you get.**

- A trained classifier and daily predictions for all 150 tickers
- A dashboard ranking the universe by predicted risk, with per-stock detail
- A chat assistant that answers questions about individual stocks using retrieved
  news and computed statistics

---

## 2. Architecture

```
┌──────────────┐                      ┌──────────────┐
│   yfinance   │                      │ Google News  │
│  OHLCV, 5y   │                      │     RSS      │
│  150 tickers │                      │  by company  │
└──────┬───────┘                      └──────┬───────┘
       │                                     │
       ▼                                     ▼
┌──────────────────────────────────────────────────────┐
│                    INGESTION                          │
│   raw/prices/{ticker}/{date}.parquet                  │
│   raw/news/{ticker}/{date}.json                       │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│                  ETL  (PySpark)                       │
│                                                       │
│   clean_transform.py                                  │
│     clean, standardise, LEFT JOIN prices + news       │
│                    ↓  processed/base/                 │
│   sentiment.py                                        │
│     VADER compound score per headline, 3-day average  │
│                    ↓  processed/sentiment/            │
│   feature_engineering.py                              │
│     52 features + cross-sectional risk label          │
│                    ↓  processed/features/             │
└───────────────────────┬──────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌───────────────┐ ┌─────────────┐ ┌──────────────┐
│   train.py    │ │ predict.py  │ │build_index.py│
│  4 candidates │ │ latest row  │ │  FAISS index │
│  best by F1   │ │ per ticker  │ │  per ticker  │
│   → models/   │ │→predictions/│ │→ vectorstore/│
└───────┬───────┘ └──────┬──────┘ └──────┬───────┘
        │                │               │
        └────────────────┼───────────────┘
                         ▼
┌──────────────────────────────────────────────────────┐
│              DASHBOARD  (Streamlit)                   │
│   Overview  │  Stock detail  │  Chat (FAISS + Groq)   │
└──────────────────────────────────────────────────────┘
```

**Everything persists to S3.** Nothing touches local disk — all reads and writes
round-trip through in-memory buffers in `src/utils/s3_io.py`. That keeps the VM
stateless and means the dashboard can run anywhere with credentials.

### Tech stack

| Layer | Technology |
|---|---|
| Ingestion | yfinance, Google News RSS |
| Storage | AWS S3 (Hive-partitioned Parquet + raw JSON) |
| Processing | Apache Spark 3.5.1 (PySpark), Hadoop-AWS S3A |
| Sentiment | VADER |
| ML | scikit-learn, XGBoost, LightGBM |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS (`IndexFlatIP`, one index per ticker) |
| LLM | Groq (Llama 3.3 70B) |
| Dashboard | Streamlit, Plotly |
| Testing | pytest (59 tests) |

---

## 3. Data model

### S3 layout

```
s3://<bucket>/
├── raw/
│   ├── prices/{TICKER}/{date}.parquet     150 objects, 5y daily OHLCV
│   └── news/{TICKER}/{date}.json          150 objects, ~30d headlines
├── processed/
│   ├── base/ticker=X/                     prices ⋈ news
│   ├── sentiment/ticker=X/                + VADER scores
│   ├── features/ticker=X/                 + 52 features + label  (150 partitions, ~74 MB)
│   └── predictions/latest.parquet         one row per ticker
├── models/risk_model_v1/model.pkl         best estimator + scaler + encoder
└── vectorstore/{TICKER}/
    ├── index.faiss
    └── chunks.json
```

**The three `processed/` prefixes form a strict chain — each stage reads one and
writes the next.** No stage ever overwrites its own input. Spark evaluates lazily,
so an in-place `mode("overwrite")` can delete the input directory before the read
has materialised, giving either a `FileNotFoundException` or a silently truncated
table.

### Feature groups (52 total)

| Group | Features | Rationale |
|---|---|---|
| Realised volatility | `volatility_5d/10d/20d/30d/60d/90d` | Volatility clusters; 30d matches the label horizon |
| Range-based | `parkinson_vol_20d`, `garman_klass_vol_20d` | Use intraday high/low/open — 5–7× more efficient than close-to-close |
| Term structure | `vol_ratio_20_60`, `vol_ratio_20_90` | Rising short-term vol signals regime change |
| Vol-of-vol | `vol_of_vol_60d`, `vol_of_vol_ratio` | Unstable volatility is more likely to move again |
| Downside / tails | `downside_vol_20d`, `downside_ratio`, `return_skew_60d`, `return_kurt_60d`, `extreme_move_count_20d` | Risk is asymmetric; fat tails persist |
| Overnight | `overnight_gap_vol_20d`, `avg_abs_gap_20d` | News arriving out of hours shows up as gaps |
| Market regime | `market_vol_20d`, `rel_vol_20d`, `beta_60d`, `corr_market_60d` | Volatility is heavily systematic |
| Price position | `ma_ratio_20d/60d/90d`, `drawdown_from_peak`, `max_drawdown_60d`, `pct_of_52w_range`, `momentum_20d/60d` | All scale-free ratios, not price levels |
| Oscillators | `rsi_14`, `macd_norm`, `macd_signal_norm` | Normalised by price |
| Liquidity | `volume_ratio`, `amihud_illiq_20d` | Thin stocks move further per unit of order flow |
| **Cross-sectional rank** | `xs_rank_*` (9 columns) | Puts features in the same space as the rank-based label |
| **Rank persistence** | `xs_rank_mean_60d`, `xs_rank_std_60d`, `xs_rank_drift` | The property the target relies on |
| Sentiment | `daily_sentiment`, `sentiment_3d_avg`, `article_count` | Computed and displayed, but see [limitations](#10-known-limitations) |

**Every feature is scale-free** — a ratio, a rate, or a bounded index. All 150
tickers are pooled into one model with one global scaler, so raw price levels are
meaningless: MRF trades near ₹150,000 and IDEA near ₹10, and a raw `ma_20d` column
would encode *which company a row belongs to* rather than anything about its risk.

---

## 4. Results

Test period is the most recent 20% of the timeline, chronologically separated with
a 30-day embargo.

| Target definition | Features | Baseline | Accuracy | Lift |
|---|---|---|---|---|
| Absolute threshold, **random split** | 13 | — | 76.5% | **invalid — leakage** |
| Absolute threshold, chronological split | 13 | 41.6% | 48.2% | 1.16× |
| Cross-sectional rank | 19 | 33.3% | 47.7% | 1.43× |
| **Cross-sectional rank** | **52** | **33.3%** | **52.2%** | **1.57×** |

Best model: **random forest**, macro-F1 0.522.

### The leakage finding

The original 76.5% was an artefact. A random `train_test_split` on time-series data
where the label looks 30 days forward and the features are rolling windows over the
preceding 20–90 days puts near-duplicate rows on both sides of the split. Two rows
one day apart share ~59 of 60 days of their volatility window. The model was being
scored on rows it had effectively memorised.

Replacing it with a chronological split plus embargo dropped accuracy to 48.2%.
**That drop is the fix, not a regression.** Everything after it was earned back
legitimately through feature engineering and a better-posed target.

### Model comparison (52 features, 3 classes)

```
                     accuracy  f1_macro  precision_macro  recall_macro
random_forest        0.521732  0.521908         0.533203      0.521320
logistic_regression  0.521800  0.521749         0.535029      0.521354
lightgbm             0.497487  0.499685         0.505247      0.497299
xgboost              0.487368  0.489231         0.497269      0.487126
```

**Logistic regression matched random forest to four decimal places**, and both beat
the gradient-boosted models. That indicates the relationship is largely *monotone in
rank space*: once features are rank-transformed, a linear decision boundary is
approximately the right shape, and the tree ensembles spend capacity fitting
curvature that isn't there.

---

## 5. Setup

### Prerequisites

| | Version | Notes |
|---|---|---|
| OS | Linux | Tested on Ubuntu 24.04 (also runs in WSL2 / VirtualBox) |
| Python | **3.11** | Not 3.12 — PySpark 3.5.1 breaks on it (`distutils` removal) |
| Java | 11 or 17 | Spark requirement |
| RAM | 4 GB minimum | 8 GB comfortable |
| Disk | ~12 GB | torch is the bulk of it |
| AWS | S3 bucket + IAM user | Free tier is sufficient |
| Groq | API key | Free tier at console.groq.com |

News ingestion uses Google News RSS and needs **no API key**.

### 5.1 System packages

```bash
sudo apt update
sudo apt install -y openjdk-11-jdk-headless git unzip curl

echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
java -version    # expect 11.x or 17.x
```

You do **not** need a separate Spark installation — `pip install pyspark` bundles it.

### 5.2 Python 3.11

Ubuntu 24.04 ships Python 3.12, which PySpark 3.5.1 does not support:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

Do not change the system default `python3` — Ubuntu's own tooling depends on 3.12.

### 5.3 The project

```bash
git clone <repo-url> EquiRisk && cd EquiRisk
python3.11 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# CPU-only torch FIRST — saves ~2 GB of unusable CUDA libraries
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5.4 AWS

Create an S3 bucket, block all public access, and create an IAM user with
programmatic access. Attach this inline policy — note the **two different ARNs**:
`s3:ListBucket` is a bucket-level action and needs the bare ARN, while the object
actions need the `/*` suffix. Granting only the `/*` form is the most common cause
of `AccessDenied` on listing.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
      "Resource": "arn:aws:s3:::YOUR-BUCKET"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::YOUR-BUCKET/*"
    }
  ]
}
```

### 5.5 Configuration

Create `.env` in the repository root (gitignored):

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET=your-bucket-name
GROQ_API_KEY=...
```

```bash
chmod 600 .env
```

Then edit `config/config.yaml`:

```yaml
s3:
  bucket: your-bucket-name     # must match S3_BUCKET in .env

spark:
  driver_memory: 2g            # use 1500m on a 4 GB machine
```

### 5.6 Populate company names

Google News is searched by **company name**, not ticker symbol — ticker symbols are
not unique across exchanges. This one-time script fills the `company_name` column:

```bash
python scripts/enrich_ticker_names.py     # ~5 min, writes incrementally
head -3 config/nifty150_midcap.csv
```

### 5.7 Verify

```bash
python -m src.etl.spark_session    # downloads S3A jars (~275 MB) on first run
pytest -q                          # 59 tests, no S3 or Spark needed
python -c "
import sys; sys.path.insert(0,'.')
from src.utils.s3_io import list_keys
print('bucket reachable, objects:', len(list_keys('')))"
```

`0` objects is correct for a fresh bucket.

---

## 6. Running the pipeline

```bash
python scripts/run_pipeline_cli.py                 # everything, in order
python scripts/run_pipeline_cli.py --list-stages   # show stage order
python scripts/run_pipeline_cli.py --stage train   # one stage
python scripts/run_pipeline_cli.py --verbose       # DEBUG logging
```

Exit code is 0 on success, 1 on any stage failure — safe for cron.

### Stages and timings

Measured on a 4-core, 4.5 GB Ubuntu VM with a domestic connection. Most of the
elapsed time is S3 transfer, not computation.

| Stage | What it does | Time |
|---|---|---|
| `prices` | yfinance → `raw/prices/` | ~10 min |
| `news` | Google News RSS → `raw/news/` | ~4 min |
| `etl` | Clean, join → `processed/base/` | ~15 min |
| `sentiment` | VADER → `processed/sentiment/` | ~15 min |
| `features` | 52 features + label → `processed/features/` | ~18 min |
| `train` | 4 candidate models → `models/` | ~5 min |
| `predict` | Latest row per ticker → `predictions/` | ~2 min |
| `rag` | FAISS index per ticker → `vectorstore/` | ~10 min |

**Run stages individually the first time.** If one exhausts memory you find out
after five minutes rather than forty. Long silent periods are normal — Spark logs
nothing while shuffling or writing.

### Recommended first run

```bash
for s in prices news etl sentiment features train predict rag; do
    python scripts/run_pipeline_cli.py --stage $s || break
done
```

Watch the `etl` output for the coverage diagnostic:

```
NEWS JOIN: 147/150 price symbols matched (98.0%)
```

Zero overlap escalates to `ERROR` with example symbols — that failure would
otherwise be silent, producing constant-zero sentiment features with no error.

---

## 7. The dashboard

```bash
streamlit run dashboard/app.py
# in a VM, bind to all interfaces:
streamlit run dashboard/app.py --server.address 0.0.0.0
```

Three tabs:

- **Overview** — all 150 tickers ranked by predicted risk, with a heuristic
  composite score and filters
- **Stock detail** — price chart with moving averages, volatility, beta, Sharpe,
  fundamentals, and the model's prediction
- **Chat** — ask questions about a stock; retrieves from that ticker's FAISS index
  and answers via Groq

**Do not use the "Refresh Pipeline" button on a small machine.** It runs the full
pipeline inside the Streamlit process, which already has torch loaded — Spark's JVM
on top of that will exhaust a 4 GB VM. Use the CLI and restart the dashboard.

### Two different risk numbers

The dashboard shows both, and they answer different questions:

- **Predicted risk label** — the ML classifier's forecast over a fixed 30-day
  horizon, from 52 features
- **Composite risk score** — a transparent, duration-adjustable heuristic combining
  volatility, beta and sentiment, computed live

They are displayed separately on purpose. The first is a model output; the second
is an explainable summary of current conditions.

---

## 8. Repository layout

```
config/
  config.yaml                  All paths, parameters, model settings
  nifty150_midcap.csv          symbol, company_name
src/
  ingestion/
    fetch_prices.py            yfinance → S3
    fetch_news.py              Google News RSS → S3
  etl/
    spark_session.py           SparkSession with S3A wiring
    clean_transform.py         Clean, join, news coverage diagnostic
    sentiment.py               VADER scoring
    feature_engineering.py     52 features + risk label
  ml/
    train.py                   Split, train 4 candidates, save the best
    predict.py                 Live inference on the latest row per ticker
    evaluate.py                Metrics and comparison tables
  rag/
    build_index.py             Corpus assembly, chunking, FAISS index
    retriever.py               Query embedding and top-k retrieval
    llm_client.py              Groq prompt construction
  analytics/
    market_stats.py            Beta, Sharpe, annualised volatility
    risk_score.py              Heuristic composite score
    fundamentals.py            yfinance company data
  pipeline/orchestrator.py     Stage ordering and error handling
  utils/
    s3_io.py                   All S3 access (parallel partition reads)
    config.py                  Single resolved config loader
    logging_config.py          Logging setup
dashboard/app.py               Streamlit app
scripts/
  run_pipeline_cli.py          CLI entrypoint
  enrich_ticker_names.py       One-time company name lookup
  check_news_quality.py        Inspect news precision on ambiguous tickers
notebooks/                     01–07, exploration and model comparison
tests/                         59 tests
```

---

## 9. Design decisions

**Chronological split with an embargo.** The earliest 80% of the timeline trains,
the latest 20% tests, and a 30-day block is dropped between them. Without the
embargo, the final training rows carry labels computed from forward windows that
reach into the test period — the boundary itself leaks.

**Label thresholds fitted on the training period only.** Quantile cutoffs are
fitted parameters, no different from a scaler's mean. Computing them across the full
table would bake the test period's distribution into the test labels.

**Labels require a complete forward window.** Spark's `rowsBetween(1, 30)` does not
fail when fewer than 30 future rows exist — it silently computes over whatever is
there. Rows without a full window are explicitly nulled rather than labelled from a
truncated one. Those nulled tail rows are exactly what `predict.py` scores live.

**Cross-sectional labels.** Buckets come from `percent_rank` within each date rather
than fixed global thresholds. Classes stay balanced in every period by construction,
so a market regime shift cannot skew the test set — and the target becomes a
persistent stock characteristic rather than a market-regime bet.

**Cross-sectional rank features.** The label is a rank, so the features include
ranks. Without them the model must re-derive "is 0.024 high relative to today's
cross-section?" on every date from absolute numbers alone.

**One FAISS index per ticker** rather than one large index, since every query is
scoped to a single company. Faster retrieval, no cross-ticker noise.

**Google News RSS over a commercial news API.** See below.

---

## 10. Known limitations

**News sentiment contributes nothing to the model.** It is implemented, tested and
integrated end-to-end, achieving 98% ticker coverage (2,864 articles across 147 of
150 tickers). But freely available sources provide roughly 30 days of history
against five years of price data, so the sentiment features are constant across the
entire training period — LightGBM reports 49 of 52 features used, the three
sentiment columns being the exceptions. They do inform the live dashboard and the
RAG assistant. Incorporating sentiment into the model would require multi-year
archived news from a commercial provider.

**marketaux was evaluated first and rejected.** It returned 8 articles across 100
Indian midcaps (97 tickers empty), and — more seriously — resolved ticker symbols
against a US-centric namespace: a query for `ACC` returned *American Campus
Communities*, a US student-housing REIT, rather than ACC Ltd. Those rows joined
silently and would have fed a cement company's risk model sentiment derived from
American real estate news. Google News is queried by company name, which removes the
symbol namespace entirely.

**Name-based news search degrades when the company name is a sector term.** "Oil
India" attracts generic crude-price coverage; "BSE Limited" attracts stories about
every company fined *by* the exchange. Roughly 4–8 of 8 headlines are on-topic for
these; distinctive names are near-perfect. This is dilutive noise rather than
systematic error.

**MACD uses a simple moving average, not a true EMA.** Spark has no native recursive
window function. Acceptable as a technical-indicator feature, but not a faithful MACD.

**RSI uses a simple rolling average**, not Wilder's smoothing.

**Beta in the analytics module is computed against the Nifty 50** (`^NSEI`), not a
midcap index — Yahoo has no reliable single Nifty Midcap 150 ticker. The `beta_60d`
*feature* uses an equal-weighted mean of the panel itself, which is internally
consistent.

**The growth projection in the dashboard compounds historical CAGR forward.** It is
an illustration, not a forecast, and is labelled as such.

**yfinance is pinned loosely (`>=0.2.60`) on purpose.** Yahoo changes its
undocumented endpoints every few months and older releases stop working — 0.2.40
returns HTTP 429 and a `JSONDecodeError` against the current API. Recent versions
use `curl_cffi` to perform the browser-style handshake Yahoo now requires.

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run from the repository root, or `export PYTHONPATH=~/EquiRisk`.

**`AccessDenied` on `list_keys`**
`s3:ListBucket` needs the bare bucket ARN; object actions need `/*`. See
[5.4](#54-aws).

**`InvalidAccessKeyId`**
The key doesn't exist under your account. `SignatureDoesNotMatch` would mean a wrong
secret; `403 Forbidden` on `head_bucket` usually means the bucket name belongs to
someone else.

**`HTTP 429` from Yahoo**
Rate-limited. Upgrade yfinance, wait a few minutes, and keep `sleep_sec` at 2.0 in
`fetch_prices.py`. Retrying immediately extends the cooldown.

**`Read timeout on endpoint URL`**
Slow connection to S3. `s3_io.py` sets a 300-second read timeout and adaptive
retries; partitions are fetched 16-at-a-time. If it persists, reduce `max_workers`.

**A Spark stage appears hung**
It probably isn't. Writing 150 partitions to S3 produces no log output for many
minutes. Check progress from a second terminal:
```bash
python -c "
import sys; sys.path.insert(0,'.')
from src.utils.s3_io import list_keys
print(len(list_keys('processed/features/')))"
```
A count that rises — or briefly *falls*, since `mode("overwrite")` deletes before
writing — means it is working. **Do not Ctrl+C during a write**: an interrupted
overwrite can leave the path partially deleted, and the next stage would train on a
silently incomplete dataset.

**Process killed with no error**
The Linux OOM killer. Confirm with `dmesg | tail -20`, lower `driver_memory`, and
make sure the dashboard isn't running at the same time.

**`Java heap space`**
Lower `spark.driver_memory` in `config/config.yaml`.

---

## Disclaimer

This is an educational project. Nothing it produces is investment advice.
