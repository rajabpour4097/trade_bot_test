import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .config import StrategyConfig
from .data_loader import load_csv
from .mt5_loader import fetch_m1
from .engine import run_engine
from .strategy import Trade
from .legs import get_legs
from .swing_detect import validate_swing
from .fib import build_oriented_fib
from .ml_features import extract_features


def build_dataset(data: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    feats_rows = []
    trades = run_engine(data, cfg)
    # Index trades by timestamp for quick lookup of outcomes
    trade_map = {(t.timestamp, t.direction, round(t.entry, 5), round(t.sl, 5), round(t.tp, 5)): t for t in trades}

    i = max(cfg.lookback_bars, 100)
    n = len(data)
    while i < n:
        window = data.iloc[: i + 1]
        legs = get_legs(window.tail(max(cfg.lookback_bars, 100)), cfg.threshold)
        swing_type, is_swing = validate_swing(window, legs)
        if not is_swing:
            i += 1
            continue
        impulse = legs[-3]
        fib = build_oriented_fib(swing_type, impulse["start_value"], impulse["end_value"])
        ts = data.index[i]
        row = data.iloc[i]
        # Create a hypothetical trade like engine
        if swing_type == "bearish" and not cfg.sell_only:
            entry = row["close"]
            sl = max(impulse["start_value"], entry + cfg.min_sl_pips / 10000.0)
            tp = entry - abs(entry - sl) * cfg.win_ratio
            dir_ = "SELL"
        elif swing_type == "bullish" and not cfg.sell_only:
            entry = row["close"]
            sl = min(impulse["start_value"], entry - cfg.min_sl_pips / 10000.0)
            tp = entry + abs(entry - sl) * cfg.win_ratio
            dir_ = "BUY"
        elif swing_type == "bearish" and cfg.sell_only:
            entry = row["close"]
            sl = max(impulse["start_value"], entry + cfg.min_sl_pips / 10000.0)
            tp = entry - abs(entry - sl) * cfg.win_ratio
            dir_ = "SELL"
        else:
            i += 1
            continue

        # Match with executed trade outcome if exists
        key = (ts, dir_, round(float(entry), 5), round(float(sl), 5), round(float(tp), 5))
        outcome = trade_map.get(key)
        if not outcome:
            i += 1
            continue
        y = 1 if outcome.exit_reason == "TP" else 0
        feats = extract_features(data, i, swing_type, impulse, fib, cfg.lookback_bars)
        feats_rows.append({**feats, "label": y})
        i += 1
    return pd.DataFrame(feats_rows)


def main():
    ap = argparse.ArgumentParser(description="Train ML filter to improve win-rate")
    ap.add_argument("--csv", type=str, default="gpt5_backtest/data/EURUSD1.csv")
    ap.add_argument("--out", type=str, default="gpt5_backtest/results/ml_filter")
    ap.add_argument("--mt5", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = StrategyConfig(threshold=7.0, lookback_bars=140)  # شروع با تنظیمات سخت‌ترِ جواب‌داده

    # Build dataset from CSV
    df_csv = load_csv(args.csv)
    ds_csv = build_dataset(df_csv, cfg)

    # Build dataset from MT5 July/August
    s_j = datetime.fromisoformat("2025-07-01T00:00:00")
    e_j = datetime.fromisoformat("2025-07-31T23:59:00")
    s_a = datetime.fromisoformat("2025-08-01T00:00:00")
    e_a = datetime.fromisoformat("2025-08-31T23:59:00")
    df_j = fetch_m1("EURUSD", s_j, e_j)
    df_a = fetch_m1("EURUSD", s_a, e_a)
    ds_j = build_dataset(df_j, cfg)
    ds_a = build_dataset(df_a, cfg)

    ds = pd.concat([ds_csv, ds_j, ds_a], ignore_index=True)
    ds.to_csv(out / "dataset.csv", index=False)

    # Train-test split
    X = ds.drop(columns=["label"]) if not ds.empty else pd.DataFrame()
    y = ds["label"] if not ds.empty else pd.Series(dtype=int)
    if ds.empty or len(ds["label"].unique()) < 2:
        (out / "train_report.txt").write_text("Insufficient labeled data for training.", encoding="utf-8")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    (out / "train_report.txt").write_text(report, encoding="utf-8")

    # Persist model
    import joblib
    joblib.dump({"model": clf, "features": list(X.columns)}, out / "model.joblib")


if __name__ == "__main__":
    main()
