from datetime import datetime
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None


def init_mt5() -> bool:
    if mt5 is None:
        return False
    if not mt5.initialize():
        return False
    return True


def fetch_m1(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 module not available. Install MetaTrader5 or use CSV input.")
    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MT5.")
    try:
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
        if rates is None or len(rates) == 0:
            raise RuntimeError("No rates returned from MT5.")
        df = pd.DataFrame(rates)
        df.rename(columns={"time": "timestamp", "tick_volume": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("timestamp").copy()
        df.set_index("timestamp", inplace=True)
        return df
    finally:
        mt5.shutdown()
