import argparse
from pathlib import Path
import json

from .config import StrategyConfig
from .data_loader import load_csv
from .engine import run_engine, equity_curve, monthly_stats
from .ml_filter import make_acceptor
from .metrics import summarize, to_dataframe


def main():
    p = argparse.ArgumentParser(description="Run CSV backtest with custom strategy params")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=6.0)
    p.add_argument("--entry-tolerance-pips", type=float, default=2.0)
    p.add_argument("--lookback-bars", type=int, default=100)
    p.add_argument("--two-touch-705", action="store_true")
    p.add_argument("--min-sl-pips", type=float, default=2.0)
    p.add_argument("--sell-only", action="store_true")
    p.add_argument("--timeout-bars", type=int, default=300)
    p.add_argument("--ml-model", type=str, default=None, help="Path to trained model.joblib to filter trades")
    p.add_argument("--ml-thresh", type=float, default=0.8, help="Threshold for ML model filtering")
    p.add_argument("--win-ratio", type=float, default=None, help="Override RR target (e.g., 1.2, 1.5, 2.0)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_csv(args.csv)
    cfg = StrategyConfig(
        threshold=args.threshold,
        entry_tolerance_pips=args.entry_tolerance_pips,
        lookback_bars=args.lookback_bars,
        two_touch_705=args.two_touch_705,
        min_sl_pips=args.min_sl_pips,
        sell_only=args.sell_only,
        timeout_bars=args.timeout_bars,
    )
    if args.win_ratio is not None:
        cfg.win_ratio = float(args.win_ratio)
    accept = make_acceptor(args.ml_model, args.ml_thresh) if args.ml_model else None
    trades = run_engine(data, cfg, accept_trade=accept)

    trades_df = to_dataframe(trades)
    trades_path = out / "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    summary = summarize(trades)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    equity_curve(trades).to_csv(out / "equity.csv", index=False)
    monthly_stats(trades).to_csv(out / "monthly_stats.csv", index=False)

    with open(out / "diagnostics.md", "w", encoding="utf-8") as f:
        f.write("# Backtest Diagnostics\n\n")
        f.write(f"Input: {args.csv}\n\n")
        for k, v in summary.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")


if __name__ == "__main__":
    main()
