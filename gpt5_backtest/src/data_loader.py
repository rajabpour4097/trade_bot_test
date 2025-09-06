import pandas as pd
from pathlib import Path

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close"]


def _read_with_inference(p: Path) -> pd.DataFrame:
    # Try default
    try:
        return pd.read_csv(p)
    except Exception:
        pass
    # Try delimiter inference (python engine)
    try:
        return pd.read_csv(p, sep=None, engine="python")
    except Exception:
        pass
    # Try tab-separated without header
    try:
        return pd.read_csv(p, sep="\t", header=None)
    except Exception:
        raise


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = _read_with_inference(p)

    # If parser produced a single column, try tab-delimited without header
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(p, sep="\t", header=None)
        except Exception:
            pass

    # If required columns missing, assume no header and set columns by count
    if not set(REQUIRED_COLS).issubset(set(df.columns)):
        if df.shape[1] >= 5:
            # Assign: timestamp, open, high, low, close, [volume? ...]
            cols = ["timestamp", "open", "high", "low", "close"] + [f"extra{i}" for i in range(df.shape[1] - 5)]
            df.columns = cols
        else:
            # As a fallback, set first column timestamp and hope others follow
            df.columns = ["timestamp"] + [f"col{i}" for i in range(1, df.shape[1])]

    # enforce required cols
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Parse timestamp and clean
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    # Keep only necessary columns
    keep = ["timestamp", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    df = df[keep]
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    return df
