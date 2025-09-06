from typing import List, Dict
import pandas as pd

# Minimal, faithful leg detector guided by TRADING_CONFIG['threshold'] (points)
# Uses high/low breakout logic with direction persistence until a switch.

def get_legs(data: pd.DataFrame, threshold_points: float) -> List[Dict]:
    if data.empty:
        return []
    legs = []
    idx = data.index.to_list()
    start = idx[0]
    direction = None
    start_val = data.loc[start, "close"]
    last_extreme = start_val

    for i in range(1, len(data)):
        t = idx[i]
        row = data.iloc[i]
        price = row["close"]
        if direction in (None, "up"):
            # extend up if higher highs
            last_extreme = max(last_extreme, row["high"]) if direction == "up" else last_extreme
            move = (price - start_val) * 10000.0
            if direction == "up":
                # continue until clear down move beyond threshold
                if (start_val - price) * 10000.0 <= -threshold_points:
                    pass
            if move >= threshold_points and direction is None:
                direction = "up"
        if direction in (None, "down"):
            last_extreme = min(last_extreme, row["low"]) if direction == "down" else last_extreme
            move = (start_val - price) * 10000.0
            if move >= threshold_points and direction is None:
                direction = "down"
        # switch conditions
        if direction == "up":
            drawdown_pts = (last_extreme - row["low"]) * 10000.0
            if drawdown_pts >= threshold_points:
                legs.append({
                    "direction": "up",
                    "start": start,
                    "end": t,
                    "start_value": start_val,
                    "end_value": last_extreme,
                })
                # new leg starts here
                start = t
                start_val = row["close"]
                direction = "down"
                last_extreme = row["low"]
        elif direction == "down":
            retrace_pts = (row["high"] - last_extreme) * 10000.0
            if retrace_pts >= threshold_points:
                legs.append({
                    "direction": "down",
                    "start": start,
                    "end": t,
                    "start_value": start_val,
                    "end_value": last_extreme,
                })
                start = t
                start_val = row["close"]
                direction = "up"
                last_extreme = row["high"]

    # close final leg if meaningful
    if direction is not None and len(legs) == 0 or (data.index[-1] != start):
        legs.append({
            "direction": direction or "up",
            "start": start,
            "end": data.index[-1],
            "start_value": start_val,
            "end_value": last_extreme,
        })
    return legs
