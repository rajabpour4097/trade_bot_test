import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from .runner import run_backtest


def _collect_csvs(folder: Path, prefix: str, start: datetime, end: datetime) -> list[Path]:
    paths = []
    d = start
    while d <= end:
        name = f"EURUSD_{prefix}_{d.date()}.csv"
        p = folder / name
        if p.exists():
            paths.append(p)
        d += timedelta(days=1)
    return paths


def load_real_signals(signals_dir: str, start: str, end: str) -> pd.DataFrame:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    folder = Path(signals_dir)
    files = _collect_csvs(folder, "signals", s, e)
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # expected columns: dt_utc, direction, entry, sl, tp
        df["dt_utc"] = pd.to_datetime(df["dt_utc"])  # assume UTC
        df["direction"] = df["direction"].str.upper()
        df = df[["dt_utc", "direction", "entry", "sl", "tp"]]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["dt_utc","direction","entry","sl","tp"])


def load_bt_trades(trades_csv: str) -> pd.DataFrame:
    df = pd.read_csv(trades_csv)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # naive
    return df


def match_signals(real_df: pd.DataFrame, bt_df: pd.DataFrame, tolerance_minutes: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if real_df.empty or bt_df.empty:
        return pd.DataFrame(), real_df, bt_df
    # sort by time and match per direction
    tol = pd.Timedelta(minutes=tolerance_minutes)
    matches = []
    used_bt_idx = set()
    bt_sorted = bt_df.sort_values("timestamp")
    for dir_ in ["BUY", "SELL"]:
        r = real_df[real_df["direction"] == dir_].sort_values("dt_utc")
        b = bt_sorted[bt_sorted["direction"] == dir_]
        if r.empty or b.empty:
            continue
        # merge_asof does nearest backward; instead do manual nearest window search
        for ridx, rr in r.iterrows():
            # candidate window
            start_t = rr["dt_utc"] - tol
            end_t = rr["dt_utc"] + tol
            cand = b[(b["timestamp"] >= start_t) & (b["timestamp"] <= end_t)].copy()
            if cand.empty:
                continue
            cand["timedelta"] = (cand["timestamp"] - rr["dt_utc"]).abs()
            cand.sort_values("timedelta", inplace=True)
            # find first not used
            chosen = None
            for _, bb in cand.iterrows():
                idx = int(bb.name)
                if idx not in used_bt_idx:
                    chosen = bb
                    used_bt_idx.add(idx)
                    break
            if chosen is None:
                continue
            price_diff_pips = abs(float(rr["entry"]) - float(chosen["entry"])) * 10000.0
            matches.append({
                "real_time": rr["dt_utc"],
                "real_direction": dir_,
                "real_entry": rr["entry"],
                "bt_time": chosen["timestamp"],
                "bt_entry": chosen["entry"],
                "time_diff_min": float(chosen["timedelta"].total_seconds())/60.0,
                "price_diff_pips": price_diff_pips,
            })
    match_df = pd.DataFrame(matches).sort_values("real_time") if matches else pd.DataFrame(columns=["real_time","real_direction","real_entry","bt_time","bt_entry","time_diff_min","price_diff_pips"])
    matched_bt_times = set(match_df["bt_time"]) if not match_df.empty else set()
    unmatched_real = real_df[~real_df["dt_utc"].isin(match_df["real_time"])].copy() if not match_df.empty else real_df.copy()
    unmatched_bt = bt_df[~bt_df["timestamp"].isin(matched_bt_times)].copy()
    return match_df, unmatched_real, unmatched_bt


def run_validation(real_root: str, start: str, end: str, out_root: str):
    outdir_all = Path(out_root) / "ALL"
    outdir_sell = Path(out_root) / "SELL_ONLY"
    outdir_all.mkdir(parents=True, exist_ok=True)
    outdir_sell.mkdir(parents=True, exist_ok=True)

    # Run backtests
    _, _ = run_backtest(csv=None, outdir=str(outdir_all), sell_only=False,
                        mt5_symbol="EURUSD", mt5_start=start, mt5_end=end)
    _, _ = run_backtest(csv=None, outdir=str(outdir_sell), sell_only=True,
                        mt5_symbol="EURUSD", mt5_start=start, mt5_end=end)

    # Load real signals
    real_signals = load_real_signals(signals_dir=str(Path(real_root)/"signals"), start=start, end=end)

    # Load backtest trades
    bt_all = load_bt_trades(str(outdir_all/"trades.csv"))
    bt_sell = load_bt_trades(str(outdir_sell/"trades.csv"))

    # Match
    m_all, ur_all, ub_all = match_signals(real_signals, bt_all)
    m_sell, ur_sell, ub_sell = match_signals(real_signals[real_signals["direction"]=="SELL"], bt_sell)

    # Save
    m_all.to_csv(outdir_all/"matches.csv", index=False)
    ur_all.to_csv(outdir_all/"unmatched_real.csv", index=False)
    ub_all.to_csv(outdir_all/"unmatched_backtest.csv", index=False)

    m_sell.to_csv(outdir_sell/"matches.csv", index=False)
    ur_sell.to_csv(outdir_sell/"unmatched_real.csv", index=False)
    ub_sell.to_csv(outdir_sell/"unmatched_backtest.csv", index=False)

    # Summaries
    def summarize_match(mdf: pd.DataFrame) -> dict:
        if mdf.empty:
            return {"matches":0, "avg_time_diff_min": None, "median_time_diff_min": None, "avg_price_diff_pips": None, "median_price_diff_pips": None}
        return {
            "matches": int(len(mdf)),
            "avg_time_diff_min": float(mdf["time_diff_min"].mean()),
            "median_time_diff_min": float(mdf["time_diff_min"].median()),
            "avg_price_diff_pips": float(mdf["price_diff_pips"].mean()),
            "median_price_diff_pips": float(mdf["price_diff_pips"].median()),
        }

    rep = {
        "period": {"start": start, "end": end},
        "real_signals": int(len(real_signals)),
        "all": {
            "bt_trades": int(len(bt_all)),
            "unmatched_real": int(len(ur_all)),
            "unmatched_backtest": int(len(ub_all)),
            **summarize_match(m_all),
        },
        "sell_only": {
            "bt_trades": int(len(bt_sell)),
            "unmatched_real": int(len(ur_sell)),
            "unmatched_backtest": int(len(ub_sell)),
            **summarize_match(m_sell),
        },
    }
    (Path(out_root)/"validation_summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Validate backtest against real bot logs")
    parser.add_argument("--real-root", type=str, required=True, help="Path to folder containing signals/, trades_dir/, events/")
    parser.add_argument("--start", type=str, required=True, help="ISO start, e.g., 2025-09-02T00:00:00")
    parser.add_argument("--end", type=str, required=True, help="ISO end, e.g., 2025-09-05T23:59:00")
    parser.add_argument("--out", type=str, default="gpt5_backtest/results/validation_sep2_5")
    args = parser.parse_args()

    run_validation(args.real_root, args.start, args.end, args.out)


if __name__ == "__main__":
    main()
