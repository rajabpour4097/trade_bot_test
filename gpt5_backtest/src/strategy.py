from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

from .config import StrategyConfig
from .legs import get_legs
from .swing_detect import validate_swing
from .fib import build_oriented_fib


@dataclass
class Trade:
    timestamp: pd.Timestamp
    direction: str  # 'BUY' or 'SELL'
    entry: float
    sl: float
    tp: float
    rr: float
    sl_pips: float
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None  # 'TP'|'SL'|'TIMEOUT'
    exit_price: Optional[float] = None


def pips(a: float, b: float) -> float:
    return abs(a - b) * 10000.0


class SwingFibStrategy:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def generate_signals(self, data: pd.DataFrame) -> List[Trade]:
        trades: List[Trade] = []
        legs = get_legs(data.tail(max(self.cfg.lookback_bars, 100)), self.cfg.threshold)
        swing_type, is_swing = validate_swing(data, legs)
        if not is_swing:
            return trades
        last3 = legs[-3:]
        impulse = last3[0]
        # Define oriented fib along impulse
        fib = build_oriented_fib(swing_type, impulse["start_value"], impulse["end_value"])
        zone_low, zone_high = (min(fib["0.705"], fib["0.9"]), max(fib["0.705"], fib["0.9"]))
        tol = self.cfg.entry_tolerance_pips / 10000.0

        # SELL-only optional filter
        if self.cfg.sell_only and swing_type == "bullish":
            return trades

        # scan last N bars for entry touches into zone
        window = data.tail(self.cfg.lookback_bars)
        touches_705 = 0
        for ts, row in window.iterrows():
            high, low, close, open_ = row["high"], row["low"], row["close"], row["open"]
            touched = (low <= zone_high + tol and high >= zone_low - tol)
            if not touched:
                continue
            # count 0.705 touch events (within tolerance)
            near_705 = (low <= fib["0.705"] + tol and high >= fib["0.705"] - tol)
            if near_705:
                touches_705 += 1
                if self.cfg.two_touch_705 and touches_705 < 2:
                    continue
            if swing_type == "bearish":
                direction = "SELL"
                entry = close  # conservative at close
                # SL at impulse start (above), ensure min_sl
                raw_sl = max(impulse["start_value"], entry + self.cfg.min_sl_pips / 10000.0)
                sl = raw_sl
                risk = pips(entry, sl)
                tp = entry - risk * self.cfg.win_ratio / 10000.0
            else:
                direction = "BUY"
                entry = close
                raw_sl = min(impulse["start_value"], entry - self.cfg.min_sl_pips / 10000.0)
                sl = raw_sl
                risk = pips(entry, sl)
                tp = entry + risk * self.cfg.win_ratio / 10000.0
            trades.append(Trade(ts, direction, entry, sl, tp, self.cfg.win_ratio, risk))
            # one entry per swing detection for simplicity
            break
        return trades

    def simulate(self, data: pd.DataFrame, trades: List[Trade]) -> List[Trade]:
        if not trades:
            return []
        out: List[Trade] = []
        # simple bar-based fill: after entry ts, walk forward and check SL/TP hits by low/high
        for t in trades:
            after = data.loc[data.index >= t.timestamp].iloc[1:].copy()
            for idx, (ts, row) in enumerate(after.iterrows(), start=1):
                if t.direction == "SELL":
                    # SL hit if high >= sl; TP hit if low <= tp
                    if row["high"] >= t.sl:
                        t.exit_timestamp = ts
                        t.exit_reason = "SL"
                        t.exit_price = t.sl
                        break
                    if row["low"] <= t.tp:
                        t.exit_timestamp = ts
                        t.exit_reason = "TP"
                        t.exit_price = t.tp
                        break
                else:
                    if row["low"] <= t.sl:
                        t.exit_timestamp = ts
                        t.exit_reason = "SL"
                        t.exit_price = t.sl
                        break
                    if row["high"] >= t.tp:
                        t.exit_timestamp = ts
                        t.exit_reason = "TP"
                        t.exit_price = t.tp
                        break
                if self.cfg.timeout_bars and idx >= self.cfg.timeout_bars:
                    t.exit_timestamp = ts
                    t.exit_reason = "TIMEOUT"
                    t.exit_price = row["close"]
                    break
            if t.exit_timestamp is None:
                # timeout at end
                t.exit_timestamp = data.index[-1]
                t.exit_reason = "TIMEOUT"
                t.exit_price = data.iloc[-1]["close"]
            out.append(t)
        return out
