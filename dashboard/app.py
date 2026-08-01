"""
dashboard/app.py

Single-page dashboard -- NO sidebar, NO Streamlit multi-page nav. All
navigation (Overview / Company Detail / Chat Assistant) is done via
st.tabs() on this one page. If you still have dashboard/pages/*.py
files sitting there, Streamlit will keep auto-generating its own sidebar
page-nav UI just from that folder's presence -- delete or rename
dashboard/pages/ (e.g. to dashboard/_unused_pages/) so the sidebar
actually disappears; this file no longer imports or needs them.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Resolve everything relative to the repo root rather than the process's
# working directory -- `streamlit run` can be invoked from anywhere, and
# a bare open("config/config.yaml") only works if you happen to launch
# from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.analytics.fundamentals import fetch_fundamentals, format_large_number
from src.analytics.market_stats import compute_window_stats, fetch_benchmark_returns
from src.analytics.risk_score import compute_risk_score, categorize_risk_score, score_to_color
from src.pipeline.orchestrator import run_full_pipeline
from src.rag.llm_client import answer_query
from src.utils.s3_io import read_hive_partitioned_parquet_s3, read_parquet_s3, predictions_key

st.set_page_config(page_title="EquiRisk", page_icon="\U0001F4C8", layout="wide")

# ---------------------------------------------------------------------
# Theme: light/dark toggle + card styling
# ---------------------------------------------------------------------

THEMES = {
    "light": {
        "bg": "#eef2f7", "card": "#ffffff", "border": "#dbe3ec",
        "text": "#0f172a", "muted": "#64748b", "accent": "#1d4e89",
        "accent_soft": "#e3edf9", "input_bg": "#ffffff",
        "table_head": "#f1f5f9", "table_row": "#ffffff",
        "shadow": "0 1px 2px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.05)",
        "plotly": "plotly_white", "grid": "#e8edf3", "grid_filter": "none",
    },
    "dark": {
        "bg": "#0d1117", "card": "#161b22", "border": "#2a313c",
        "text": "#e6edf3", "muted": "#8b949e", "accent": "#58a6ff",
        "accent_soft": "#1c2b3d", "input_bg": "#0f141a",
        "table_head": "#1c2128", "table_row": "#161b22",
        "shadow": "0 1px 2px rgba(0,0,0,.4), 0 4px 12px rgba(0,0,0,.3)",
        "plotly": "plotly_dark", "grid": "#262c36", "grid_filter": "none",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "light"

T = THEMES[st.session_state.theme]


def inject_css(t):
    """Page CSS plus overrides for Streamlit's own widgets.

    The dataframe, multiselect and buttons carry baked-in colours that ignore
    page-level rules, which is why a partial theme leaves dark widgets sitting
    on a light page. Each is targeted by its stable data-testid or baseweb
    attribute rather than by generated class names.
    """
    st.markdown(f"""
    <style>
      .stApp, .main, section.main {{ background:{t['bg']} !important; }}
      .block-container {{ padding-top:2rem; max-width:1400px; }}

      h1,h2,h3,h4,h5,h6,p,span,label,li,
      .stMarkdown, .stMarkdown p,
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {{
          color:{t['text']} !important;
      }}
      [data-testid="stCaptionContainer"] p {{ color:{t['muted']} !important; }}

      div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"],
      div[data-testid="stMetric"], div[data-testid="stExpander"] {{
          background:{t['card']} !important;
          border:1px solid {t['border']} !important;
          border-radius:16px !important;
          padding:16px 18px !important;
          box-shadow:{t['shadow']} !important;
          margin-bottom:18px !important;
      }}
      div[data-testid="stMetric"] label p {{
          color:{t['muted']} !important; font-size:.82rem !important;
          text-transform:uppercase; letter-spacing:.04em;
      }}
      div[data-testid="stMetricValue"] {{ color:{t['text']} !important; font-weight:700 !important; }}

      div[data-testid="stDataFrame"] > div {{
          background:{t['table_row']} !important; border-radius:10px !important; overflow:hidden;
      }}
      div[data-testid="stDataFrame"] * {{ color:{t['text']} !important; }}
      div[data-testid="stDataFrame"] [role="columnheader"] {{
          background:{t['table_head']} !important; font-weight:600 !important;
          border-bottom:2px solid {t['border']} !important;
      }}
      div[data-testid="stDataFrame"] [role="gridcell"] {{
          background:{t['table_row']} !important;
          border-bottom:1px solid {t['border']} !important;
      }}
      canvas {{ background:{t['table_row']} !important; }}

      div[data-baseweb="select"] > div {{
          background:{t['input_bg']} !important;
          border:1px solid {t['border']} !important;
          border-radius:10px !important; color:{t['text']} !important;
      }}
      div[data-baseweb="select"] span, div[data-baseweb="select"] input {{ color:{t['text']} !important; }}
      span[data-baseweb="tag"] {{
          background:{t['accent_soft']} !important; color:{t['accent']} !important;
          border:1px solid {t['accent']}44 !important;
          border-radius:8px !important; font-weight:600 !important;
      }}
      span[data-baseweb="tag"] svg {{ fill:{t['accent']} !important; }}
      ul[data-baseweb="menu"], div[data-baseweb="popover"] div {{
          background:{t['card']} !important; color:{t['text']} !important;
      }}

      .stTextInput input, .stNumberInput input, .stTextArea textarea {{
          background:{t['input_bg']} !important; color:{t['text']} !important;
          border:1px solid {t['border']} !important; border-radius:10px !important;
      }}

      .stButton button, .stDownloadButton button {{
          background:{t['card']} !important; color:{t['text']} !important;
          border:1px solid {t['border']} !important; border-radius:10px !important;
          font-weight:600 !important; box-shadow:{t['shadow']} !important;
      }}
      .stButton button p {{ color:inherit !important; }}
      .stButton button:hover {{ border-color:{t['accent']} !important; color:{t['accent']} !important; }}

      .stTabs [data-baseweb="tab-list"] {{ gap:8px; background:transparent;
          border-bottom:1px solid {t['border']}; }}
      .stTabs [data-baseweb="tab"] {{
          background:{t['card']} !important; border:1px solid {t['border']} !important;
          border-bottom:none !important; border-radius:12px 12px 0 0 !important;
          padding:10px 20px !important;
      }}
      .stTabs [data-baseweb="tab"] p {{ color:{t['muted']} !important; font-weight:600; }}
      .stTabs [aria-selected="true"] {{ background:{t['accent_soft']} !important;
          border-bottom:3px solid {t['accent']} !important; }}
      .stTabs [aria-selected="true"] p {{ color:{t['accent']} !important; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background:transparent !important; }}

      div[data-testid="stToggle"] label p {{ color:{t['text']} !important; }}
      section[data-testid="stSidebar"] > div {{ background:{t['card']} !important; }}
      div[data-testid="stVerticalBlock"] {{ gap:.55rem !important; }}
      hr {{ border-color:{t['border']} !important; margin:1rem 0 !important; }}


      /* ---- chat input: renders its own dark container + red focus ring ---- */
      div[data-testid="stChatInput"],
      div[data-testid="stChatInput"] > div,
      div[data-testid="stChatInput"] div[data-baseweb="textarea"],
      div[data-testid="stChatInput"] div[data-baseweb="base-input"] {{
          background:{t['input_bg']} !important;
          border-color:{t['border']} !important;
          border-radius:12px !important;
          box-shadow:none !important;
      }}
      div[data-testid="stChatInput"] textarea {{
          background:{t['input_bg']} !important;
          color:{t['text']} !important;
          caret-color:{t['accent']} !important;
      }}
      div[data-testid="stChatInput"] textarea::placeholder {{
          color:{t['muted']} !important; opacity:1 !important;
      }}
      div[data-testid="stChatInput"] button svg,
      div[data-testid="stChatInput"] svg {{ fill:{t['accent']} !important; }}
      div[data-testid="stChatInput"]:focus-within,
      div[data-testid="stChatInput"] div:focus-within {{
          border-color:{t['accent']} !important;
          box-shadow:0 0 0 2px {t['accent']}33 !important;
      }}
      div[data-testid="stChatMessage"] {{
          background:{t['card']} !important;
          border:1px solid {t['border']} !important;
          border-radius:14px !important;
      }}

      /* ---- dataframe draws into a canvas that ignores CSS colours ---- */
      div[data-testid="stDataFrame"] iframe {{
          background:{t['table_row']} !important;
          border-radius:10px !important;
          filter:{t['grid_filter']};
      }}

      /* ---- slider: default accent is red ---- */
      .stSlider div[data-baseweb="slider"] div[role="slider"] {{
          background:{t['accent']} !important; border-color:{t['accent']} !important;
      }}
      .stSlider div[data-baseweb="slider"] > div > div > div:first-child {{
          background:{t['accent']} !important;
      }}
      .stSlider [data-testid="stTickBar"],
      .stSlider [data-testid="stTickBarMin"],
      .stSlider [data-testid="stTickBarMax"],
      .stSlider [data-testid="stThumbValue"] {{
          color:{t['muted']} !important;
      }}
      .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {{
          color:{t['accent']} !important; font-weight:600 !important;
      }}

      /* ---- catch-all for remaining red accents ---- */
      [data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {{
          border-color:{t['accent']} !important;
      }}


      /* ---- the "Running..." status widget renders its own dark bar ---- */
      div[data-testid="stStatusWidget"],
      div[data-testid="stStatusWidget"] > div {{
          background:{t['card']} !important;
          border:1px solid {t['border']} !important;
          border-radius:10px !important;
          box-shadow:{t['shadow']} !important;
      }}
      div[data-testid="stStatusWidget"] label,
      div[data-testid="stStatusWidget"] span {{ color:{t['text']} !important; }}
      div[data-testid="stStatusWidget"] code {{
          background:{t['accent_soft']} !important; color:{t['accent']} !important;
      }}


      /* Streamlit's status widget renders its own dark bar in the top-right
         while cached functions execute. It ignores page CSS and only ever
         shows a transient loading state, so it is hidden rather than themed. */
      div[data-testid="stStatusWidget"] {{ display:none !important; }}
      #MainMenu {{ visibility:hidden; }}
      footer {{ visibility:hidden; }}
      header[data-testid="stHeader"] {{ background:transparent !important; }}

      .equirisk-card {{ background:{t['card']}; border:1px solid {t['border']};
          border-radius:16px; padding:18px 20px; box-shadow:{t['shadow']}; margin-bottom:18px; }}
    </style>
    """, unsafe_allow_html=True)


def style_fig(fig, t=None):
    """Theme a Plotly figure so it sits inside its card rather than as a
    bright rectangle on a dark page."""
    t = t or T
    fig.update_layout(
        template=t["plotly"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], size=12), title_font=dict(color=t["text"], size=15),
        margin=dict(t=46, b=26, l=14, r=14),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=t["text"])),
        hoverlabel=dict(bgcolor=t["card"], font_color=t["text"]),
    )
    fig.update_xaxes(gridcolor=t["grid"], linecolor=t["border"], tickfont=dict(color=t["muted"]))
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["border"], tickfont=dict(color=t["muted"]))
    return fig


inject_css(T)




PREDICTIONS_KEY = predictions_key()

ML_LABEL_COLORS = {"Low": "#22c55e", "Medium": "#eab308", "High": "#ea2626"}


# ---------------------------------------------------------------------
# Cached data loaders -- all Yahoo/S3 calls funnel through these so
# reruns (every widget interaction) don't refetch from the network.
# ---------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=300)
def load_full_table() -> pd.DataFrame:
    config = _load_config()
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["paths"]["processed_features"]
    try:
        return read_hive_partitioned_parquet_s3(prefix, bucket, partition_col="ticker")
    except RuntimeError:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    """The ML-classifier's live prediction for each ticker's current
    (null-label) row -- see src/ml/predict.py. The feature table's own
    risk_label is null for the most recent row per ticker BY DESIGN
    (it's a forward-looking label with no future data yet) -- this
    predictions table is what actually has a usable "current risk" per
    company, and both the Overview and Company Detail tabs read it."""
    config = _load_config()
    bucket = config["s3"]["bucket"]
    try:
        return read_parquet_s3(PREDICTIONS_KEY, bucket)
    except Exception:
        return pd.DataFrame(columns=["ticker", "date", "predicted_risk_label", "predicted_at"])


@st.cache_data(ttl=3600)
def get_benchmark_returns_cached(benchmark_ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Fetches the benchmark index ONCE per hour per date range, shared
    across every ticker/duration combination in this session -- this is
    what actually fixes beta showing N/A: without this cache, switching
    the duration dropdown or the company selector re-hits Yahoo live on
    every single rerun, which gets rate-limited/blocked fast."""
    return fetch_benchmark_returns(benchmark_ticker, start_date, end_date)


@st.cache_data(ttl=86400)
def cached_fundamentals(ticker_ns: str) -> dict:
    return fetch_fundamentals(ticker_ns)


def run_refresh_pipeline():
    with st.status("Running full pipeline...", expanded=True) as status_box:
        def _callback(msg: str):
            status_box.write(msg)

        result = run_full_pipeline(status_callback=_callback)

        if result.overall_success:
            status_box.update(label="Pipeline complete", state="complete")
            st.cache_data.clear()
        else:
            status_box.update(label="Pipeline failed -- see details below", state="error")
            for stage in result.stages:
                if not stage.success:
                    st.error(f"{stage.name}: {stage.message}")


def compute_cagr(ticker_df: pd.DataFrame) -> float:
    df = ticker_df.sort_values("date")
    if len(df) < 2:
        return float("nan")
    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    if years <= 0 or start_price <= 0:
        return float("nan")
    return (end_price / start_price) ** (1 / years) - 1


def price_chart(df: pd.DataFrame, ticker: str, months_back: int) -> go.Figure:
    cutoff = df["date"].max() - pd.Timedelta(days=months_back * 30)
    windowed = df[df["date"] >= cutoff]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=windowed["date"], y=windowed["close"], name="Close", line=dict(color="#2563eb")))
    if "ma_20d" in windowed.columns:
        fig.add_trace(go.Scatter(x=windowed["date"], y=windowed["ma_20d"], name="20d MA",
                                  line=dict(color="#f59e0b", dash="dot")))
    fig.update_layout(title=f"{ticker} -- Price", height=380)
    return style_fig(fig)


def render_risk_badge(score: float, category: str, sublabel: str = ""):
    """Colored badge: continuous green->yellow->red background driven
    by score_to_color(), with the category name and numeric score on
    top of it. This replaces the old plain-text colored caption."""
    color = score_to_color(score)
    score_text = f"{score:+.0f}" if not (score is None or np.isnan(score)) else "N/A"
    st.markdown(
        f"""
        <div style="background:{color}; padding:10px 18px; border-radius:10px;
                    display:inline-block; color:#0b0f19; font-weight:700; font-size:1.05rem;">
            {category} &nbsp;·&nbsp; Score: {score_text}
        </div>
        <div style="color:{T["muted"]}; font-size:0.85rem; margin-top:4px;">{sublabel}</div>
        """,
        unsafe_allow_html=True,
    )


def render_ml_label_badge(label: str):
    color = ML_LABEL_COLORS.get(label, "#6b7280")
    st.markdown(
        f"""<span style="background:{color}; color:#0b0f19; padding:3px 10px;
             border-radius:6px; font-weight:600; font-size:0.85rem;">{label}</span>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------

config = _load_config()
analytics_config = config["analytics"]

st.title("EquiRisk")
st.caption("Risk prediction dashboard for Nifty150 midcap companies")

top_col1, top_col2, top_col3 = st.columns([4, 1, 1])
with top_col2:
    _dark = st.session_state.theme == "dark"
    if st.toggle("Dark mode", value=_dark, key="theme_toggle") != _dark:
        st.session_state.theme = "light" if _dark else "dark"
        st.rerun()
with top_col3:
    if st.button("\U0001F504 Refresh Pipeline", use_container_width=True):
        run_refresh_pipeline()

full_df = load_full_table()
predictions_df = load_predictions()

if full_df.empty:
    st.warning("No processed data available yet. Click 'Refresh Pipeline' above to get started.")
    st.stop()

tickers = sorted(full_df["ticker"].unique().tolist())

tab_overview, tab_detail, tab_chat = st.tabs(["\U0001F4CA Overview", "\U0001F3E2 Company Detail", "\U0001F4AC Chat Assistant"])

# ---------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------
with tab_overview:
    st.subheader("All Companies -- Current Risk")

    latest = full_df.sort_values("date").groupby("ticker").tail(1)
    snapshot = latest.merge(
        predictions_df[["ticker", "predicted_risk_label"]], on="ticker", how="left"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies tracked", len(snapshot))
    col2.metric("High risk", int((snapshot["predicted_risk_label"] == "High").sum()))
    col3.metric("Medium risk", int((snapshot["predicted_risk_label"] == "Medium").sum()))
    col4.metric("Low risk", int((snapshot["predicted_risk_label"] == "Low").sum()))

    st.divider()

    risk_filter = st.multiselect(
        "Filter by risk level", options=["Low", "Medium", "High"], default=["Low", "Medium", "High"],
        key="overview_risk_filter",
    )
    filtered = snapshot[snapshot["predicted_risk_label"].isin(risk_filter) | snapshot["predicted_risk_label"].isna()]

    display_df = filtered[["ticker", "date", "close", "predicted_risk_label", "volatility_20d", "sentiment_3d_avg"]].copy()
    display_df = display_df.rename(columns={
        "ticker": "Ticker", "date": "As of", "close": "Last Close",
        "predicted_risk_label": "Risk", "volatility_20d": "20d Volatility",
        "sentiment_3d_avg": "Sentiment (3d avg)",
    })
    display_df["Risk"] = display_df["Risk"].fillna("Pending")

    # st.dataframe renders into a canvas-based grid whose colours ignore page
    # CSS, so it stayed dark inside a light card. Styler output is plain HTML
    # and inherits the theme correctly.
    # Wrapped in a fixed-height scroll container -- the Styler table has no
    # height limit of its own, so 150 rows would run down the whole page.
    # Headers stay pinned via position:sticky.
    st.markdown(
        f'<div style="max-height:460px; overflow-y:auto; border-radius:10px; '
        f'border:1px solid {T["border"]};">' +
        display_df.style
            .hide(axis="index")
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background", T["table_head"]), ("color", T["text"]),
                    ("font-weight", "600"), ("padding", "10px 12px"),
                    ("text-align", "left"), ("position", "sticky"),
                    ("top", "0"), ("z-index", "2"),
                    ("border-bottom", f"2px solid {T['border']}")]},
                {"selector": "td", "props": [
                    ("color", T["text"]), ("padding", "9px 12px"),
                    ("border-bottom", f"1px solid {T['border']}")]},
                {"selector": "table", "props": [
                    ("width", "100%"), ("border-collapse", "collapse"),
                    ("font-size", "0.9rem")]},
                {"selector": "tbody tr:hover td", "props": [
                    ("background", T["accent_soft"])]},
            ])
            .to_html() + '</div>',
        unsafe_allow_html=True,
    )
    if predictions_df.empty:
        st.info("No live predictions found yet -- run `python -m src.ml.predict` (or Refresh Pipeline) to populate current risk labels.")

# ---------------------------------------------------------------------
# Tab 2: Company Detail
# ---------------------------------------------------------------------
with tab_detail:
    col_company, col_period = st.columns([2, 2])
    with col_company:
        default_idx = tickers.index(st.session_state.get("selected_ticker", tickers[0])) if st.session_state.get("selected_ticker") in tickers else 0
        selected_ticker = st.selectbox("Company", tickers, index=default_idx, key="detail_company")
    with col_period:
        period_label = st.selectbox("Stats duration", list(analytics_config["periods"].keys()), index=2, key="detail_period")

    st.session_state["selected_ticker"] = selected_ticker

    ticker_df = full_df[full_df["ticker"] == selected_ticker].sort_values("date")
    window_days = analytics_config["periods"][period_label]

    bench_start = str(full_df["date"].min())[:10]
    bench_end = str(full_df["date"].max())[:10]
    benchmark_returns = get_benchmark_returns_cached(analytics_config["benchmark_ticker"], bench_start, bench_end)

    with st.spinner(f"Computing {period_label} stats..."):
        stats = compute_window_stats(
            ticker_df, window_days, benchmark_returns, analytics_config["risk_free_rate_annual"]
        )
        recent_window = ticker_df.tail(window_days)
        avg_sentiment = recent_window["sentiment_3d_avg"].mean() if "sentiment_3d_avg" in recent_window.columns else 0.0
        risk_score = compute_risk_score(stats["volatility"], stats["beta"], avg_sentiment)
        risk_category = categorize_risk_score(risk_score)

    if benchmark_returns.empty:
        st.warning(
            f"Couldn't fetch benchmark data ({analytics_config['benchmark_ticker']}) -- "
            "beta will show N/A until this resolves (usually a temporary Yahoo Finance rate-limit)."
        )
    elif np.isnan(stats["beta"]):
        st.info(
            f"Benchmark data ({analytics_config['benchmark_ticker']}) was fetched, but there wasn't enough "
            "overlapping trading-day data with this ticker's window to compute beta reliably -- "
            "check terminal logs for the overlap count, or try a longer duration."
        )

    st.subheader(f"{selected_ticker} -- {period_label} snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"\u20B9{stats['current_price']:.2f}" if not np.isnan(stats["current_price"]) else "N/A")
    c2.metric("Beta", f"{stats['beta']:.2f}" if not np.isnan(stats["beta"]) else "N/A")
    c3.metric("Sharpe Ratio", f"{stats['sharpe']:.2f}" if not np.isnan(stats["sharpe"]) else "N/A")
    c4.metric("Volatility (ann.)", f"{stats['volatility']*100:.1f}%" if not np.isnan(stats["volatility"]) else "N/A")

    render_risk_badge(risk_score, risk_category, sublabel=f"Composite score for the {period_label} window (-100 very low \u2192 +100 very high)")

    pred_row = predictions_df[predictions_df["ticker"] == selected_ticker]
    if not pred_row.empty:
        st.write("")
        st.write("ML classifier's current label:")
        render_ml_label_badge(pred_row["predicted_risk_label"].iloc[0])

    st.divider()

    st.subheader("Price History")
    months_back = st.slider(
        "Chart range (months)", min_value=1,
        max_value=analytics_config["price_chart_max_years"] * 12,
        value=analytics_config["price_chart_default_months"],
        key="detail_months_back",
    )
    st.plotly_chart(style_fig(price_chart(ticker_df, selected_ticker, months_back)), use_container_width=True)

    st.divider()

    st.subheader("Financial Health")
    ticker_ns = f"{selected_ticker}{config['tickers']['suffix']}"
    with st.spinner("Fetching fundamentals..."):
        fundamentals = cached_fundamentals(ticker_ns)

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("P/E Ratio", f"{fundamentals['pe_ratio']:.1f}" if fundamentals["pe_ratio"] else "N/A")
    f2.metric("Net Income", format_large_number(fundamentals["net_income"]))
    f3.metric("Profit Margin", f"{fundamentals['profit_margin']*100:.1f}%" if fundamentals["profit_margin"] else "N/A")
    f4.metric("Debt-to-Equity", f"{fundamentals['debt_to_equity']:.1f}" if fundamentals["debt_to_equity"] else "N/A")
    f5.metric("Market Cap", format_large_number(fundamentals["market_cap"]))
    if fundamentals.get("sector"):
        st.caption(f"Sector: {fundamentals['sector']}")
    st.caption("Fundamentals via Yahoo Finance -- cached up to 24h, independent of the pipeline refresh.")

    st.divider()

    st.subheader("Investment Growth Calculator")
    st.caption(
        "Projects a future value by compounding this stock's own historical CAGR. "
        "A projection based on the past, NOT a guarantee or recommendation."
    )

    cagr = compute_cagr(ticker_df)
    calc_col1, calc_col2, calc_col3 = st.columns(3)
    with calc_col1:
        invest_amount = st.number_input("Investment amount (\u20B9)", min_value=1000, value=100000, step=1000, key="calc_amount")
    with calc_col2:
        invest_years = st.selectbox("Time period", [1, 2, 3, 5, 10], index=2, key="calc_years")
    with calc_col3:
        st.metric("Historical CAGR", f"{cagr*100:.1f}%" if not np.isnan(cagr) else "N/A")

    if not np.isnan(cagr):
        projected_value = invest_amount * ((1 + cagr) ** invest_years)
        gain = projected_value - invest_amount
        # Colour and wording follow the sign. st.success() was hardcoded, so a
        # projected LOSS was rendered on a green background and still labelled
        # a "gain" -- with a minus sign inside the number.
        is_gain = gain >= 0
        bg     = "#dcfce7" if is_gain else "#fee2e2"
        fg     = "#166534" if is_gain else "#991b1b"
        border = "#86efac" if is_gain else "#fca5a5"
        word   = "gain" if is_gain else "loss"
        st.markdown(
            f"""<div style="background:{bg}; color:{fg}; border:1px solid {border};
                 border-radius:10px; padding:12px 16px; margin:6px 0;">
              Projected value after {invest_years} year(s):
              <b>\u20B9{projected_value:,.0f}</b>
              ({word} of <b>\u20B9{abs(gain):,.0f}</b>, assuming the historical CAGR holds)
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("Not enough price history to compute a CAGR for this ticker yet.")

# ---------------------------------------------------------------------
# Tab 3: Chat Assistant
# ---------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask about a company's risk profile")
    st.caption("Answers are grounded in retrieved news and computed stats via RAG + Groq.")

    default_idx = tickers.index(st.session_state.get("selected_ticker", tickers[0])) if st.session_state.get("selected_ticker") in tickers else 0
    chat_ticker = st.selectbox("Company", tickers, index=default_idx, key="chat_company")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}
    if chat_ticker not in st.session_state["chat_history"]:
        st.session_state["chat_history"][chat_ticker] = []

    history = st.session_state["chat_history"][chat_ticker]
    for role, message in history:
        with st.chat_message(role):
            st.write(message)

    user_query = st.chat_input(f"Ask about {chat_ticker}'s risk profile...")
    if user_query:
        history.append(("user", user_query))
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_query(chat_ticker, user_query)
                st.write(answer)
        history.append(("assistant", answer))