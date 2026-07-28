# EquiRisk

**Equity risk classification and RAG-powered analysis for the Nifty Midcap 150.**

EquiRisk ingests five years of daily OHLCV data and recent news for 150 Indian
midcap stocks, processes it through a Spark ETL pipeline on S3, trains a
classifier to predict each stock's forward 30-day volatility bucket
(Low / Medium / High), and serves the results through a Streamlit dashboard with
a retrieval-augmented chat assistant.

Built as a CDAC capstone project.

---

## Architecture

```
                    ┌──────────────┐   ┌──────────────┐
                    │  yfinance    │   │ Google News  │
                    │  OHLCV, 5y   │   │     RSS      │
                    └──────┬───────┘   └──────┬───────┘
                           │                  │
                    ┌──────▼──────────────────▼───────┐
   INGESTION       │  raw/prices/    raw/news/        │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  clean_transform.py             │
   ETL              │  clean, standardise, join       │
   (PySpark)        │            ↓ processed/base/    │
                    │  sentiment.py    (VADER)        │
                    │            ↓ processed/sentiment/│
                    │  feature_engineering.py         │
                    │  returns, volatility, MA, RSI,  │
                    │  MACD, forward risk label       │
                    │            ↓ processed/features/│
                    └──────────────┬──────────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                 │
        ┌────────▼──────┐ ┌────────▼──────┐ ┌───────▼────────┐
   ML   │  train.py     │ │  predict.py   │ │  build_index.py│  RAG
        │  4 candidates │ │  live scoring │ │  FAISS/ticker  │
        │  → models/    │ │  → predictions│ │  → vectorstore/│
        └────────┬──────┘ └────────┬──────┘ └───────┬────────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
   DASHBOARD        │  Streamlit: Overview │ Detail │ │
                    │  Chat (FAISS retrieval + Groq)  │
                    └─────────────────────────────────┘
```

Everything persists to S3. Nothing is written to local disk — all reads and
writes round-trip through in-memory buffers in `src/utils/s3_io.py`.

---

## Tech stack

| Layer | Technology |
|---|---|
| Ingestion | yfinance, Google News RSS |
| Storage | AWS S3 (partitioned Parquet + raw JSON) |
| Processing | Apache Spark 3.5 (PySpark), Hadoop-AWS S3A |
| Sentiment | VADER |
| ML | scikit-learn, XGBoost, LightGBM |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS (`IndexFlatIP`, one index per ticker) |
| LLM | Groq (Llama 3.3 70B) |
| Dashboard | Streamlit, Plotly |

---

## Repository layout

```
config/
  config.yaml              All paths, parameters, model settings
  nifty150_midcap.csv      Ticker master list
src/
  ingestion/               yfinance + Google News RSS → S3 (raw)
  etl/                     Spark: clean, join, sentiment, features
  ml/                      Training, evaluation, live inference
  rag/                     Corpus building, FAISS retrieval, Groq client
  analytics/               Beta, Sharpe, composite risk score, fundamentals
  pipeline/orchestrator.py Stage ordering and error handling
  utils/                   S3 I/O, config loading, logging
dashboard/app.py           Single-page Streamlit app (3 tabs)
notebooks/                 01–07, exploration and model comparison
scripts/run_pipeline_cli.py
tests/
```

---

## Setup

**Prerequisites:** Python 3.11, JDK 11 (for Spark), an AWS account with an S3
bucket, and a [Groq](https://console.groq.com/) API key. News ingestion uses
Google News RSS and needs no key.

```bash
git clone <repo-url> && cd EquiRisk
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repository root (it is gitignored):

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET=equirisk-data
GROQ_API_KEY=...
```

Then set your bucket name in `config/config.yaml` under `s3.bucket`.

---

## Running the pipeline

```bash
python scripts/run_pipeline_cli.py                 # full pipeline
python scripts/run_pipeline_cli.py --list-stages   # show stage order
python scripts/run_pipeline_cli.py --stage train   # re-run one stage
python scripts/run_pipeline_cli.py --verbose       # DEBUG logging
```

Stages run in order and stop at the first failure. Exit code is 0 on success,
1 otherwise, so this is safe to schedule via cron.

```bash
streamlit run dashboard/app.py
```

The dashboard's **Refresh Pipeline** button calls the same
`run_full_pipeline()` as the CLI — the two cannot drift apart.

```bash
pytest                                             # from the repo root
```

---

## Design decisions worth knowing about

**The train/test split is chronological, not random.** The label looks 30
trading days forward and the features are rolling windows over the preceding
20–90 days, so two rows one day apart share nearly all of their underlying
information. A random split scatters those near-duplicate rows across train and
test, and the model gets credit for recognising rows it has effectively already
seen. The earliest 80% of the timeline trains, the latest 20% tests, and a
30-day embargo block is dropped between them so no training row's forward label
window reaches into the test period. Reported accuracy is lower than a random
split would give, and it is the number that actually means something.

**Risk-label thresholds are fitted on the training period only.** The quantile
cutoffs defining Low/Medium/High are themselves fitted parameters. Computing
them across the full table would bake the test period's volatility distribution
into the test labels.

**Labels require a complete forward window.** Spark's `rowsBetween(1, 30)` does
not fail when fewer than 30 future rows exist — it quietly computes over
whatever is there. Rows without a full window are explicitly nulled rather than
labelled from a truncated one. Those nulled tail rows are exactly what
`predict.py` scores live.

**Each ETL stage reads one S3 prefix and writes the next.** Spark evaluates
lazily, so `df.write.mode("overwrite")` to the path a DataFrame was read from
can clear the input directory before the read has materialised. The chain is
`base → sentiment → features`, never in place.

**Two different risk numbers, on purpose.** The ML classifier predicts a
discrete Low/Medium/High class over a fixed 30-day forward horizon. The
dashboard's composite risk score (`analytics/risk_score.py`) is a transparent,
duration-adjustable heuristic combining volatility, beta and sentiment. They
answer different questions and are displayed separately.

**One FAISS index per ticker** rather than one large index, since every query in
this dashboard is scoped to a single company. Keeps retrieval fast and avoids
cross-ticker noise.

---

## Known limitations

- **MACD uses a simple moving average, not a true EMA.** Spark has no native
  recursive window function. Fine as a technical-indicator feature; not a
  faithful MACD.
- **RSI uses a simple rolling average**, not Wilder's smoothing.
- **Beta is computed against the Nifty 50** (`^NSEI`), not a midcap index —
  Yahoo has no reliable single Nifty Midcap 150 ticker for this purpose.
- **The risk-free rate is a flat config constant**, not a live G-Sec yield feed.
- **News is recent-only.** Google News RSS returns roughly the last 30 days,
  not five years, so sentiment features are dense for recent rows and null for
  most of the history. The model is therefore driven mainly by technicals over
  the full training period. ETL logs a `NEWS JOIN` line on every run reporting
  the coverage rate; if it is low, treat the sentiment features as noise.

- **marketaux was evaluated first and rejected.** It returned 8 articles across
  100 Indian midcaps (97 tickers empty), and — more seriously — resolved ticker
  symbols against a US-centric namespace: a query for `ACC` returned American
  Campus Communities, a US student-housing REIT, rather than ACC Ltd. Those rows
  joined silently and would have fed a cement company's risk model sentiment
  derived from American real estate news. Google News is queried by company
  name, which removes the symbol namespace entirely.
- **The growth projection in the dashboard compounds historical CAGR forward.**
  It is an illustration, not a forecast, and is labelled as such.

---

## Disclaimer

This is an educational project. Nothing it produces is investment advice.
