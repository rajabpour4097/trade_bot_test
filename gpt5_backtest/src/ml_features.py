from __future__ import annotations
from typing import Dict, Any
import math
import pandas as pd


def _safe(val: float) -> float:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return 0.0
    return float(val)


def extract_features(
    data: pd.DataFrame,
    idx: int,
    swing_type: str,
    impulse: Dict[str, Any],
    fib: Dict[str, float],
    lookback_bars: int,
) -> Dict[str, float]:
    """Compute simple, fast features at decision bar idx.
    Assumes data has columns: open, high, low, close.
    """
    row = data.iloc[idx]
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    open_ = float(row["open"])
    # Fib zone
    zone_low = min(fib["0.705"], fib["0.9"])
    zone_high = max(fib["0.705"], fib["0.9"])
    zone_mid = (zone_low + zone_high) / 2.0
    zone_width_pips = abs(zone_high - zone_low) * 10000.0
    entry_vs_mid_pips = (close - zone_mid) * 10000.0
    # Candle stats
    body = abs(close - open_) * 10000.0
    upper_wick = max(0.0, high - max(close, open_)) * 10000.0
    lower_wick = max(0.0, min(close, open_) - low) * 10000.0
    # Impulse metrics
    imp_size_pips = abs(impulse["end_value"] - impulse["start_value"]) * 10000.0
    # Rolling volatility (std of returns last 30 bars)
    start_look = max(0, idx - min(lookback_bars, 200))
    win = data.iloc[start_look: idx + 1]
    ret = win["close"].pct_change().dropna()
    vol_std = float(ret.std()) if len(ret) else 0.0
    # Time features
    ts = data.index[idx]
    minute_of_day = ts.hour * 60 + ts.minute
    tod_sin = math.sin(2 * math.pi * minute_of_day / (24 * 60))
    tod_cos = math.cos(2 * math.pi * minute_of_day / (24 * 60))

    feats = {
        "zone_width_pips": _safe(zone_width_pips),
        "entry_vs_mid_pips": _safe(entry_vs_mid_pips),
        "candle_body_pips": _safe(body),
        "upper_wick_pips": _safe(upper_wick),
        "lower_wick_pips": _safe(lower_wick),
        "impulse_size_pips": _safe(imp_size_pips),
        "vol_std": _safe(vol_std),
        "tod_sin": _safe(tod_sin),
        "tod_cos": _safe(tod_cos),
        "is_bullish_swing": 1.0 if swing_type == "bullish" else 0.0,
        "is_bearish_swing": 1.0 if swing_type == "bearish" else 0.0,
    }
    return feats
