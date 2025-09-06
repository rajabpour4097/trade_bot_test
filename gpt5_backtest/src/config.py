from dataclasses import dataclass

@dataclass
class StrategyConfig:
    threshold: float = 6.0            # points threshold for legs
    fib_705: float = 0.705
    fib_90: float = 0.9
    entry_tolerance_pips: float = 2.0
    min_swing_size: int = 4
    win_ratio: float = 1.2
    min_sl_pips: float = 2.0
    lookback_bars: int = 100
    sell_only: bool = False
    two_touch_705: bool = False       # require two touches near 0.705 before entry
    timeout_bars: int = 300           # max bars to keep a trade open

@dataclass
class RunConfig:
    symbol: str = "EURUSD"
    risk_pct: float = 0.01
    commission_per_lot_side: float = 0.0
    max_daily_trades: int = 10
    session_iran_start: str = "09:00"
    session_iran_end: str = "21:00"
    use_session_filter: bool = True
    timezone: str = "Asia/Tehran"
