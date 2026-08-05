"""
src/analytics/risk_score.py

Computes a presentation-layer composite risk score on a -100 (very
low risk) to +100 (very high risk) scale, bucketed into 5 categories,
for a given duration's window_stats (from market_stats.py) plus average
sentiment over the same window.

This is DELIBERATELY separate from the ML classifier's Low/Medium/High
risk_label -- that model predicts a discrete class from a fixed 30-day
forward horizon; this score is a transparent, duration-adjustable
heuristic combining volatility, beta, and sentiment, meant for the
"quick glance" dashboard metric. Document both explicitly as different
things -- they answer different questions (a trained classifier's
forecast vs. a hand-weighted composite of current conditions).

Weights and clipping ranges below are a reasonable starting point, not
a tuned/validated model -- adjust them if the resulting scores don't
match your own intuition once you see real numbers for the 150 tickers.
"""

import numpy as np

VOLATILITY_CLIP_MAX = 0.80   # annualized volatility of 80%+ treated as max risk
BETA_CLIP_MIN = 0.0
BETA_CLIP_MAX = 2.5

WEIGHTS = {
    "volatility": 0.5,
    "beta": 0.3,
    "sentiment": 0.2,
}

CATEGORY_THRESHOLDS = [
    (-100, -60, "Very Low Risk"),
    (-60, -20, "Low Risk"),
    (-20, 20, "Neutral Risk"),
    (20, 60, "High Risk"),
    (60, 100, "Very High Risk"),
]


def _clip_scale(value: float, lo: float, hi: float) -> float:
    """Clips value to [lo, hi] then rescales to 0-100."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0  # neutral default when a component is missing
    clipped = max(lo, min(hi, value))
    return (clipped - lo) / (hi - lo) * 100.0


def compute_risk_score(volatility: float, beta: float, avg_sentiment: float) -> float:
    """Returns a score from -100 to +100.

    volatility: annualized volatility (e.g. 0.35 = 35%)
    beta: computed vs benchmark index
    avg_sentiment: mean VADER compound score over the window, -1 to +1
    """
    vol_score = _clip_scale(volatility, 0, VOLATILITY_CLIP_MAX)
    beta_score = _clip_scale(beta, BETA_CLIP_MIN, BETA_CLIP_MAX)

    sentiment_risk_score = _clip_scale(-avg_sentiment, -1, 1)

    composite_0_100 = (
        WEIGHTS["volatility"] * vol_score
        + WEIGHTS["beta"] * beta_score
        + WEIGHTS["sentiment"] * sentiment_risk_score
    )

    return round((composite_0_100 - 50) * 2, 1)


def categorize_risk_score(score: float) -> str:
    """Maps a -100..100 score to one of the 5 named risk categories."""
    if score is None or np.isnan(score):
        return "Unknown"
    for lo, hi, label in CATEGORY_THRESHOLDS:
        if lo <= score <= hi:
            return label
    return "Unknown"


def score_to_color(score: float) -> str:
    """Maps a -100..100 score to a hex color on a continuous green (-100)
    -> yellow (0) -> red (+100) gradient, for a colored badge/bar in the
    dashboard. Returns a neutral gray for a missing/NaN score."""
    if score is None or np.isnan(score):
        return "#6b7280"  # gray -- unknown

    s = max(-100.0, min(100.0, score))
    if s <= 0:
        t = (s + 100) / 100  # 0 at -100 -> 1 at 0
        r = int(34 + t * (234 - 34))
        g = int(150 + t * (179 - 150))
        b = int(74 + t * (8 - 74))
    else:
        t = s / 100  # 0 at 0 -> 1 at +100
        r = int(234 + t * (220 - 234))
        g = int(179 + t * (38 - 179))
        b = int(8 + t * (38 - 8))
    return f"#{r:02x}{g:02x}{b:02x}"