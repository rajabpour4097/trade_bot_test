import pandas as pd
from typing import List
from .strategy import Trade, pips


def summarize(trades: List[Trade]) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "net_r": 0.0,
            "buy_win_rate": 0.0,
            "sell_win_rate": 0.0,
        }
    wins = sum(1 for t in trades if t.exit_reason == "TP")
    losses = sum(1 for t in trades if t.exit_reason == "SL")
    timeouts = sum(1 for t in trades if t.exit_reason == "TIMEOUT")
    total = len(trades)
    wr = wins / total * 100.0
    # R accounting: +rr on TP, -1 on SL, 0 on TIMEOUT
    r_values = [(t.rr if t.exit_reason == "TP" else (-1.0 if t.exit_reason == "SL" else 0.0)) for t in trades]
    avg_r = sum(r_values) / total
    net_r = sum(r_values)
    # directional breakdown
    buys = [t for t in trades if t.direction == "BUY"]
    sells = [t for t in trades if t.direction == "SELL"]
    buy_wr = (sum(1 for t in buys if t.exit_reason == "TP") / len(buys) * 100.0) if buys else 0.0
    sell_wr = (sum(1 for t in sells if t.exit_reason == "TP") / len(sells) * 100.0) if sells else 0.0
    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wr,
        "avg_r": avg_r,
        "net_r": net_r,
        "buy_win_rate": buy_wr,
        "sell_win_rate": sell_wr,
    }


def to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "timestamp","direction","entry","sl","tp","rr","sl_pips","exit_timestamp","exit_reason","exit_price"
        ])
    rows = []
    for t in trades:
        rows.append({
            "timestamp": t.timestamp,
            "direction": t.direction,
            "entry": t.entry,
            "sl": t.sl,
            "tp": t.tp,
            "rr": t.rr,
            "sl_pips": t.sl_pips,
            "exit_timestamp": t.exit_timestamp,
            "exit_reason": t.exit_reason,
            "exit_price": t.exit_price,
        })
    df = pd.DataFrame(rows)
    return df
