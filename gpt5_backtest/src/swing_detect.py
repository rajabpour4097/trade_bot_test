from typing import Tuple, List, Dict
import pandas as pd


def validate_swing(data: pd.DataFrame, legs: List[Dict]) -> Tuple[str, bool]:
    """Return (swing_type, is_swing) using the 3-leg pattern and 3 opposite-color candles rule.
    This mirrors swing.py intent while being explicit.
    """
    if len(legs) < 3:
        return "", False
    last3 = legs[-3:]
    # Up swing pattern: impulse up, pullback down
    if last3[1]["end_value"] > last3[0]["start_value"] and last3[0]["end_value"] > last3[1]["end_value"]:
        # Count bearish candles within pullback leg window
        try:
            s_idx = data.index.get_loc(last3[1]["start"])
        except KeyError:
            s_idx = int(data.index.get_indexer([last3[1]["start"]], method="nearest")[0])
        try:
            e_idx = data.index.get_loc(last3[1]["end"])
        except KeyError:
            e_idx = int(data.index.get_indexer([last3[1]["end"]], method="nearest")[0])
        true_candles, first_flag = 0, False
        for k in range(s_idx, e_idx + 1):
            row = data.iloc[k]
            if row["close"] < row["open"]:
                true_candles += 2 if first_flag else 1
                first_flag = True
        if true_candles >= 3:
            return "bullish", True
    # Down swing pattern: impulse down, pullback up
    if last3[1]["end_value"] < last3[0]["start_value"] and last3[0]["end_value"] < last3[1]["end_value"]:
        try:
            s_idx = data.index.get_loc(last3[1]["start"])
        except KeyError:
            s_idx = int(data.index.get_indexer([last3[1]["start"]], method="nearest")[0])
        try:
            e_idx = data.index.get_loc(last3[1]["end"])
        except KeyError:
            e_idx = int(data.index.get_indexer([last3[1]["end"]], method="nearest")[0])
        true_candles, first_flag = 0, False
        for k in range(s_idx, e_idx + 1):
            row = data.iloc[k]
            if row["close"] > row["open"]:
                true_candles += 2 if first_flag else 1
                first_flag = True
        if true_candles >= 3:
            return "bearish", True
    return "", False
