import argparse
import json
from pathlib import Path
from datetime import datetime
from itertools import product

import pandas as pd

from .config import StrategyConfig
from .mt5_loader import fetch_m1
from .engine import run_engine
from .metrics import to_dataframe
from .validate_real import load_real_signals, match_signals


def score_config(real_df: pd.DataFrame, bt_df: pd.DataFrame, tolerance_minutes: int = 30) -> dict:
    m, ur, ub = match_signals(real_df, bt_df, tolerance_minutes=tolerance_minutes)
    matches = 0 if m is None or m.empty else len(m)
    median_td = None if m is None or m.empty else float(m["time_diff_min"].median())
    median_pd = None if m is None or m.empty else float(m["price_diff_pips"].median())
    return {
        "matches": int(matches),
        "unmatched_real": int(0 if real_df is None else len(ur)),
        "unmatched_backtest": int(0 if bt_df is None else len(ub)),
        "median_time_diff_min": median_td,
        "median_price_diff_pips": median_pd,
        "matches_df": m,
        "unmatched_real_df": ur,
        "unmatched_backtest_df": ub,
    }


def run_tuning(real_root: str, start: str, end: str, out_root: str,
               tolerance_minutes: int = 30, mode: str = "fast", early_stop: bool = False,
               time_offsets: list[int] | None = None):
    outdir = Path(out_root)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load real signals once
    real_df = load_real_signals(signals_dir=str(Path(real_root) / "signals"), start=start, end=end)

    # Fetch MT5 data once
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    data = fetch_m1("EURUSD", s, e)

    # Parameter grid
    if mode == "wide":
        thresholds = [5.0, 6.0, 7.0, 8.0]
        entry_tols = [1.0, 1.5, 2.0, 3.0]
        lookbacks = [80, 100, 120, 160]
        two_touches = [False, True]
        min_sls = [2.0, 3.0]
        sell_only_opts = [False, True]
    elif mode == "fast":
        thresholds = [6.0, 7.0]
        entry_tols = [2.0, 3.0]
        lookbacks = [100, 140]
        two_touches = [False, True]
        min_sls = [2.0]
        sell_only_opts = [False, True]
    else:  # micro
        thresholds = [6.0]
        entry_tols = [2.0, 3.0]
        lookbacks = [120]
        two_touches = [False, True]
        min_sls = [2.0]
        sell_only_opts = [False, True]

    leaderboard = []
    best = None
    best_artifacts = None

    # Comparator: matches desc, median_time_diff asc, unmatched_real asc
    def is_better(a: dict, b: dict | None) -> bool:
        if b is None:
            return True
        if a["matches"] != b["matches"]:
            return a["matches"] > b["matches"]
        a_td = float("inf") if a["median_time_diff_min"] is None else a["median_time_diff_min"]
        b_td = float("inf") if b["median_time_diff_min"] is None else b["median_time_diff_min"]
        if a_td != b_td:
            return a_td < b_td
        return a["unmatched_real"] < b["unmatched_real"]

    # Time offsets (minutes) to compensate timezone mismatch between real logs and MT5 UTC
    if time_offsets is None:
        time_offsets = [0]
    total = len(time_offsets) * len(thresholds) * len(entry_tols) * len(lookbacks) * len(two_touches) * len(min_sls) * len(sell_only_opts)
    idx = 0
    stop_all = False
    for off in time_offsets:
        if off == 0:
            shifted_real = real_df
        else:
            shifted_real = real_df.copy()
            if not shifted_real.empty:
                shifted_real["dt_utc"] = pd.to_datetime(shifted_real["dt_utc"]) + pd.Timedelta(minutes=off)
        for th, tol, lb, tt, msl, so in product(thresholds, entry_tols, lookbacks, two_touches, min_sls, sell_only_opts):
            idx += 1
            cfg = StrategyConfig(
                threshold=th,
                entry_tolerance_pips=tol,
                lookback_bars=lb,
                two_touch_705=tt,
                min_sl_pips=msl,
                sell_only=so,
            )
            trades = run_engine(data, cfg)
            bt_df = to_dataframe(trades)

            # Full-direction matching against real signals
            sc = score_config(shifted_real, bt_df, tolerance_minutes=tolerance_minutes)
            row = {
                "idx": idx,
                "time_offset_min": off,
                "threshold": th,
                "entry_tolerance_pips": tol,
                "lookback_bars": lb,
                "two_touch_705": tt,
                "min_sl_pips": msl,
                "sell_only": so,
                "matches": sc["matches"],
                "unmatched_real": sc["unmatched_real"],
                "unmatched_backtest": sc["unmatched_backtest"],
                "median_time_diff_min": sc["median_time_diff_min"],
                "median_price_diff_pips": sc["median_price_diff_pips"],
            }
            leaderboard.append(row)

            if is_better(row, best):
                best = row
                best_artifacts = {
                    "cfg": cfg,
                    "bt_df": bt_df.copy(),
                    "matches_df": sc["matches_df"].copy() if sc["matches_df"] is not None else pd.DataFrame(),
                    "unmatched_real_df": sc["unmatched_real_df"].copy() if sc["unmatched_real_df"] is not None else pd.DataFrame(),
                    "unmatched_backtest_df": sc["unmatched_backtest_df"].copy() if sc["unmatched_backtest_df"] is not None else pd.DataFrame(),
                }
                if early_stop and best.get("matches", 0) and best["matches"] > 0:
                    stop_all = True
                    break
        if stop_all:
            break
            if b is None:
                return True
            if a["matches"] != b["matches"]:
                return a["matches"] > b["matches"]
            # tie break on time diff (None treated as inf)
            a_td = float("inf") if a["median_time_diff_min"] is None else a["median_time_diff_min"]
            b_td = float("inf") if b["median_time_diff_min"] is None else b["median_time_diff_min"]
            if a_td != b_td:
                return a_td < b_td
            return a["unmatched_real"] < b["unmatched_real"]

            if is_better(row, best):
                best = row
                best_artifacts = {
                    "cfg": cfg,
                    "bt_df": bt_df.copy(),
                    "matches_df": sc["matches_df"].copy() if sc["matches_df"] is not None else pd.DataFrame(),
                    "unmatched_real_df": sc["unmatched_real_df"].copy() if sc["unmatched_real_df"] is not None else pd.DataFrame(),
                    "unmatched_backtest_df": sc["unmatched_backtest_df"].copy() if sc["unmatched_backtest_df"] is not None else pd.DataFrame(),
                }
                if early_stop and best.get("matches", 0) and best["matches"] > 0:
                    break

    # Save leaderboard and best artifacts
    lb_df = pd.DataFrame(leaderboard).sort_values(
        ["matches", "median_time_diff_min", "unmatched_real"], ascending=[False, True, True]
    )
    lb_path_csv = outdir / "leaderboard.csv"
    lb_df.to_csv(lb_path_csv, index=False)
    lb_path_json = outdir / "leaderboard.json"
    lb_path_json.write_text(json.dumps(lb_df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    best_dir = outdir / "best"
    best_dir.mkdir(exist_ok=True)
    # Config
    cfg_path = best_dir / "best_config.json"
    cfg_dict = {
        "threshold": best["threshold"],
        "entry_tolerance_pips": best["entry_tolerance_pips"],
        "lookback_bars": best["lookback_bars"],
        "two_touch_705": best["two_touch_705"],
        "min_sl_pips": best["min_sl_pips"],
        "sell_only": best["sell_only"],
        "time_offset_min": best.get("time_offset_min", 0),
    }
    cfg_path.write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    # Trades and matches
    best_artifacts["bt_df"].to_csv(best_dir / "trades.csv", index=False)
    best_artifacts["matches_df"].to_csv(best_dir / "matches.csv", index=False)
    best_artifacts["unmatched_real_df"].to_csv(best_dir / "unmatched_real.csv", index=False)
    best_artifacts["unmatched_backtest_df"].to_csv(best_dir / "unmatched_backtest.csv", index=False)

    # Summary
    summary = {
        "period": {"start": start, "end": end},
        "tolerance_minutes": tolerance_minutes,
        "grid_size": total,
        "best": best,
        "leaderboard_top": lb_df.head(10).to_dict(orient="records"),
    }
    (outdir / "tuning_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Parameter tuning to maximize match vs real signals")
    parser.add_argument("--real-root", type=str, required=True, help="Root folder with signals/")
    parser.add_argument("--start", type=str, required=True, help="ISO start, e.g., 2025-09-02T00:00:00")
    parser.add_argument("--end", type=str, required=True, help="ISO end, e.g., 2025-09-05T23:59:00")
    parser.add_argument("--out", type=str, default="gpt5_backtest/results/param_tuning")
    parser.add_argument("--tolerance-minutes", type=int, default=30)
    parser.add_argument("--mode", type=str, choices=["micro","fast","wide"], default="fast")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--time-offsets", type=str, default="0,180,-180",
                        help="Comma-separated minute offsets to apply to real signals timestamps for matching")
    args = parser.parse_args()

    try:
        offsets = [int(x) for x in args.time_offsets.split(",") if x.strip()]
    except Exception:
        offsets = [0]
    run_tuning(args.real_root, args.start, args.end, args.out,
               tolerance_minutes=args.tolerance_minutes, mode=args.mode, early_stop=args.early_stop,
               time_offsets=offsets)


if __name__ == "__main__":
    main()
