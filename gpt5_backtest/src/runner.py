import argparse
import json
from pathlib import Path

from .config import StrategyConfig, RunConfig
from .data_loader import load_csv
from .mt5_loader import fetch_m1
from .strategy import SwingFibStrategy
from .metrics import summarize, to_dataframe
from .engine import run_engine, equity_curve, monthly_stats
from .ml_filter import make_acceptor


def run_backtest(csv: str = None, outdir: str = "gpt5_backtest/results", sell_only: bool = False,
                 mt5_symbol: str = None, mt5_start: str = None, mt5_end: str = None,
                 strategy_cfg: StrategyConfig | None = None,
                 ml_model: str | None = None,
                 ml_thresh: float = 0.8,
                 win_ratio: float | None = None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if csv:
        data = load_csv(csv)
    elif mt5_symbol and mt5_start and mt5_end:
        from datetime import datetime
        s = datetime.fromisoformat(mt5_start)
        e = datetime.fromisoformat(mt5_end)
        df = fetch_m1(mt5_symbol, s, e)
        data = df.copy()
        # also persist a copy for reproducibility
        (out / "input_data.csv").write_text(df.to_csv(index=False))
    else:
        raise ValueError("Provide either --csv or all of --mt5-symbol, --mt5-start, --mt5-end")

    # Build strategy config (allow override for tuning)
    s_cfg = strategy_cfg if strategy_cfg is not None else StrategyConfig(sell_only=sell_only)
    if win_ratio is not None:
        s_cfg.win_ratio = float(win_ratio)
    # Use the sequential engine for parity with a live loop
    accept = make_acceptor(ml_model, ml_thresh) if ml_model else None
    trades = run_engine(data, s_cfg, accept_trade=accept)

    # Save trades and summary
    trades_df = to_dataframe(trades)
    trades_path = out / "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    summary = summarize(trades)
    summary_path = out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Equity and monthly stats
    eq = equity_curve(trades)
    eq_path = out / "equity.csv"
    eq.to_csv(eq_path, index=False)
    mon = monthly_stats(trades)
    mon_path = out / "monthly_stats.csv"
    mon.to_csv(mon_path, index=False)

    # Diagnostics markdown
    diag_path = out / "diagnostics.md"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("# Backtest Diagnostics\n\n")
        f.write(f"Input: {csv}\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: {v}\n")
        f.write("\nFiles saved:\n")
        f.write(f"- trades: {trades_path}\n")
        f.write(f"- summary: {summary_path}\n")
        f.write(f"- equity: {eq_path}\n")
        f.write(f"- monthly: {mon_path}\n")

    return str(trades_path), str(summary_path)


def main():
    parser = argparse.ArgumentParser(description="Run GPT5 backtest")
    parser.add_argument("--csv", type=str, help="Path to CSV with timestamp,open,high,low,close,volume")
    parser.add_argument("--out", type=str, default="gpt5_backtest/results", help="Output directory")
    parser.add_argument("--sell-only", action="store_true", help="Enable SELL-only filter")
    parser.add_argument("--mt5-symbol", type=str, help="MT5 symbol to fetch (alternative to --csv)")
    parser.add_argument("--mt5-start", type=str, help="Start ISO datetime, e.g., 2025-08-01T00:00:00")
    parser.add_argument("--mt5-end", type=str, help="End ISO datetime, e.g., 2025-08-31T23:59:00")
    parser.add_argument("--ml-model", type=str, help="Path to trained model.joblib to filter trades", default=None)
    parser.add_argument("--ml-thresh", type=float, help="Probability threshold to accept a trade [0-1]", default=0.8)
    parser.add_argument("--win-ratio", type=float, help="Override RR target (e.g., 1.2, 1.5, 2.0)")
    args = parser.parse_args()

    run_backtest(csv=args.csv, outdir=args.out, sell_only=args.sell_only,
                 mt5_symbol=args.mt5_symbol, mt5_start=args.mt5_start, mt5_end=args.mt5_end,
                 strategy_cfg=None, ml_model=args.ml_model, ml_thresh=args.ml_thresh, win_ratio=args.win_ratio)


if __name__ == "__main__":
    main()
