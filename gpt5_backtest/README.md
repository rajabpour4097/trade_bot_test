# GPT5 Backtest

A faithful backtest of your MT5 swing/Fibonacci strategy with CSV/MT5 data support and detailed analytics.

## Features
- Strategy parity: legs → swing → oriented Fibonacci (0.705–0.9) → entry/SL/TP → staged management (configurable).
- Data sources: CSV with `timestamp,open,high,low,close,volume` or direct MT5 M1 download.
- Outputs: trades.csv, equity.csv, summary.json, monthly_stats.json, diagnostics.md with BUY/SELL breakdowns.
- CLI runner for quick runs, parameters, and filters (e.g., SELL-only).

## Quick start
- Place a CSV under `gpt5_backtest/data` or use the `--mt5` flag to pull M1 from MT5.
- Run the CLI to generate results under `gpt5_backtest/results`.

## Validate
In the next step, we’ll compare these results with your real bot’s trade logs to sanity-check the backtest.
