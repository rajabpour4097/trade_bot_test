from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import pandas as pd
import pytz

from .config import StrategyConfig, RunConfig
from .swing_detect import validate_swing
from .legs import get_legs
from .fib import build_oriented_fib
from .strategy import Trade, pips


def in_session(ts: pd.Timestamp, run_cfg: RunConfig) -> bool:
    if not run_cfg.use_session_filter:
        return True
    # Try to convert to Iran time if tz-aware; otherwise assume it's already local or UTC but compare hours only
    hour_min = ts.strftime("%H:%M")
    return (run_cfg.session_iran_start <= hour_min <= run_cfg.session_iran_end)


def _bar_touches_zone(row: pd.Series, low: float, high: float, tol: float) -> bool:
    return (row["low"] <= high + tol) and (row["high"] >= low - tol)


def _near_level(row: pd.Series, level: float, tol: float) -> bool:
    return (row["low"] <= level + tol) and (row["high"] >= level - tol)


def simulate_from(data: pd.DataFrame, start_idx: int, t: Trade, timeout_bars: int) -> Tuple[Trade, int]:
    # Walk forward from start_idx+1 to resolve exit; return updated trade and last index used
    for step, (ts, row) in enumerate(list(data.iloc[start_idx + 1:].iterrows()), start=1):
        if t.direction == "SELL":
            if row["high"] >= t.sl:
                t.exit_timestamp = ts
                t.exit_reason = "SL"
                t.exit_price = t.sl
                return t, start_idx + step
            if row["low"] <= t.tp:
                t.exit_timestamp = ts
                t.exit_reason = "TP"
                t.exit_price = t.tp
                return t, start_idx + step
        else:
            if row["low"] <= t.sl:
                t.exit_timestamp = ts
                t.exit_reason = "SL"
                t.exit_price = t.sl
                return t, start_idx + step
            if row["high"] >= t.tp:
                t.exit_timestamp = ts
                t.exit_reason = "TP"
                t.exit_price = t.tp
                return t, start_idx + step
        if timeout_bars and step >= timeout_bars:
            t.exit_timestamp = ts
            t.exit_reason = "TIMEOUT"
            t.exit_price = row["close"]
            return t, start_idx + step
    # Fell off the end
    t.exit_timestamp = data.index[-1]
    t.exit_reason = "TIMEOUT"
    t.exit_price = data.iloc[-1]["close"]
    return t, len(data) - 1


def run_engine(
    data: pd.DataFrame,
    s_cfg: StrategyConfig,
    run_cfg: Optional[RunConfig] = None,
    accept_trade: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> List[Trade]:
    run_cfg = run_cfg or RunConfig()
    trades: List[Trade] = []
    tol = s_cfg.entry_tolerance_pips / 10000.0

    i = max(s_cfg.lookback_bars, 100)
    n = len(data)
    while i < n:
        ts = data.index[i]
        if not in_session(ts, run_cfg):
            i += 1
            continue

        window = data.iloc[: i + 1]
        legs = get_legs(window.tail(max(s_cfg.lookback_bars, 100)), s_cfg.threshold)
        swing_type, is_swing = validate_swing(window, legs)
        if not is_swing:
            i += 1
            continue
        impulse = legs[-3]
        fib = build_oriented_fib(swing_type, impulse["start_value"], impulse["end_value"])
        zone_low, zone_high = (min(fib["0.705"], fib["0.9"]), max(fib["0.705"], fib["0.9"]))

        row = data.iloc[i]
        if not _bar_touches_zone(row, zone_low, zone_high, tol):
            i += 1
            continue

        # SELL-only option
        if s_cfg.sell_only and swing_type == "bullish":
            i += 1
            continue

        # Two-touch 0.705 confirmation within lookback window
        if s_cfg.two_touch_705:
            touches = 0
            look = data.iloc[max(0, i - s_cfg.lookback_bars) : i + 1]
            for _, r in look.iterrows():
                if _near_level(r, fib["0.705"], tol):
                    touches += 1
            if touches < 2:
                i += 1
                continue

        # Optional ML/logic filter before committing the trade
        if accept_trade is not None:
            ctx: Dict[str, Any] = {
                "data": data,
                "index": i,
                "swing_type": swing_type,
                "impulse": impulse,
                "fib": fib,
                "config": s_cfg,
            }
            try:
                if not accept_trade(ctx):
                    i += 1
                    continue
            except Exception:
                # Fail-safe: if filter errors, skip filtering
                pass

        # Build trade (entry at close of current bar)
        entry = row["close"]
        if swing_type == "bearish":
            direction = "SELL"
            sl = max(impulse["start_value"], entry + s_cfg.min_sl_pips / 10000.0)
            risk = pips(entry, sl)
            tp = entry - risk * s_cfg.win_ratio / 10000.0
        else:
            direction = "BUY"
            sl = min(impulse["start_value"], entry - s_cfg.min_sl_pips / 10000.0)
            risk = pips(entry, sl)
            tp = entry + risk * s_cfg.win_ratio / 10000.0

        t = Trade(timestamp=ts, direction=direction, entry=entry, sl=sl, tp=tp, rr=s_cfg.win_ratio, sl_pips=risk)
        t, exit_idx = simulate_from(data, i, t, s_cfg.timeout_bars)
        trades.append(t)

        # Move pointer forward to after exit to avoid overlapping entries
        i = exit_idx + 1
    return trades


def equity_curve(trades: List[Trade]) -> pd.DataFrame:
    # R-based equity: +rr on TP, -1 on SL, 0 otherwise
    eq = 0.0
    rows = []
    for t in trades:
        r = t.rr if t.exit_reason == "TP" else (-1.0 if t.exit_reason == "SL" else 0.0)
        eq += r
        rows.append({
            "timestamp": t.exit_timestamp or t.timestamp,
            "equity_r": eq,
            "last_trade_r": r,
            "exit_reason": t.exit_reason,
            "direction": t.direction,
        })
    return pd.DataFrame(rows)


def monthly_stats(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["year","month","trades","wins","losses","timeouts","win_rate","net_r"])
    df = pd.DataFrame([
        {
            "month": (t.exit_timestamp or t.timestamp).strftime("%Y-%m"),
            "r": (t.rr if t.exit_reason == "TP" else (-1.0 if t.exit_reason == "SL" else 0.0)),
            "w": 1 if t.exit_reason == "TP" else 0,
            "l": 1 if t.exit_reason == "SL" else 0,
            "o": 1 if t.exit_reason == "TIMEOUT" else 0,
        }
        for t in trades
    ])
    grp = df.groupby("month").agg(trades=("r","count"), wins=("w","sum"), losses=("l","sum"), timeouts=("o","sum"), net_r=("r","sum")).reset_index()
    grp["win_rate"] = grp["wins"] / grp["trades"] * 100.0
    return grp
