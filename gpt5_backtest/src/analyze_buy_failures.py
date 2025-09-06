import argparse
from pathlib import Path
import json
import pandas as pd

from .config import StrategyConfig
from .data_loader import load_csv
from .legs import get_legs
from .swing_detect import validate_swing
from .fib import build_oriented_fib
from .engine import simulate_from
from .strategy import Trade, pips


def analyze_buys_on_csv(csv_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_csv(csv_path)

    cfg = StrategyConfig()  # use defaults; we only analyze BUY logic symmetry
    tol = cfg.entry_tolerance_pips / 10000.0
    i = max(cfg.lookback_bars, 100)
    n = len(data)

    rows = []
    while i < n:
        window = data.iloc[: i + 1]
        legs = get_legs(window.tail(max(cfg.lookback_bars, 100)), cfg.threshold)
        swing_type, is_swing = validate_swing(window, legs)
        if not is_swing:
            i += 1
            continue
        impulse = legs[-3]
        fib = build_oriented_fib(swing_type, impulse["start_value"], impulse["end_value"])
        zone_low, zone_high = (min(fib["0.705"], fib["0.9"]), max(fib["0.705"], fib["0.9"]))

        ts = data.index[i]
        row = data.iloc[i]
        touched = (row["low"] <= zone_high + tol and row["high"] >= zone_low - tol)
        if not touched:
            i += 1
            continue

        # Build trade candidate at close
        entry = row["close"]
        if swing_type == "bullish":
            direction = "BUY"
            sl_struct = impulse["start_value"]
            # current SL placement rule in engine (possibly asymmetric):
            sl = min(sl_struct, entry - cfg.min_sl_pips / 10000.0)
            risk = pips(entry, sl)
            tp = entry + risk * cfg.win_ratio / 10000.0
            t = Trade(timestamp=ts, direction=direction, entry=entry, sl=sl, tp=tp, rr=cfg.win_ratio, sl_pips=risk)
            t, exit_idx = simulate_from(data, i, t, cfg.timeout_bars)
            below_struct = sl < sl_struct
            zone_mid = (zone_low + zone_high) / 2.0
            rows.append({
                "timestamp": ts,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit_timestamp": t.exit_timestamp,
                "exit_reason": t.exit_reason,
                "sl_struct": sl_struct,
                "sl_below_struct": below_struct,
                "entry_minus_low_pips": (entry - min(row["low"], sl_struct)) * 10000.0,
                "zone_width_pips": abs(zone_high - zone_low) * 10000.0,
                "entry_vs_zone_mid_pips": (entry - zone_mid) * 10000.0,
                "impulse_start": impulse["start"],
                "impulse_end": impulse["end"],
            })
            # Skip ahead past this trade exit to avoid overlapping
            i = exit_idx + 1
            continue
        else:
            # We only analyze BUY failures; skip and move forward
            i += 1
            continue

    df = pd.DataFrame(rows)
    df.to_csv(out / "buy_trades_analysis.csv", index=False)

    # Aggregates
    total = len(df)
    wins = int((df["exit_reason"] == "TP").sum()) if total else 0
    losses = int((df["exit_reason"] == "SL").sum()) if total else 0
    timeouts = int((df["exit_reason"] == "TIMEOUT").sum()) if total else 0
    below_struct_share = float((df["sl_below_struct"].mean()) if total else 0.0)
    avg_zone_width = float(df["zone_width_pips"].mean()) if total else None
    avg_entry_vs_mid = float(df["entry_vs_zone_mid_pips"].mean()) if total else None

    rep = {
        "buy_trades": total,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": (wins / total * 100.0) if total else 0.0,
        "sl_below_swing_low_share": below_struct_share,
        "avg_zone_width_pips": avg_zone_width,
        "avg_entry_minus_zone_mid_pips": avg_entry_vs_mid,
    }
    (out / "buy_failure_summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Analyze BUY failures on CSV")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    analyze_buys_on_csv(args.csv, args.out)


if __name__ == "__main__":
    main()
