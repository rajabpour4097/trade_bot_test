from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import joblib
import pandas as pd

from .ml_features import extract_features


def load_model(model_path: str):
    p = Path(model_path)
    obj = joblib.load(p)
    model = obj.get("model", obj)
    features = obj.get("features")
    return model, features


def make_acceptor(model_path: str, proba_thresh: float = 0.8) -> Callable[[Dict[str, Any]], bool]:
    """Create an accept_trade callable using a trained classifier.

    Decision rule: predict_proba >= 0.8 for class 1 (win) to accept.
    """
    model, feature_names = load_model(model_path)

    def accept(ctx: Dict[str, Any]) -> bool:
        data = ctx["data"]
        i = ctx["index"]
        swing_type = ctx["swing_type"]
        impulse = ctx["impulse"]
        fib = ctx["fib"]
        cfg = ctx["config"]
        feats = extract_features(data, i, swing_type, impulse, fib, cfg.lookback_bars)
        # Order features as per training
        if feature_names is None:
            cols = list(feats.keys())
        else:
            cols = feature_names
        x = pd.DataFrame([[feats.get(name, 0.0) for name in cols]], columns=cols)
        proba = 0.0
        try:
            proba = float(model.predict_proba(x)[0][1])
        except Exception:
            # Fallback to decision function if no proba
            pred = float(model.predict(x)[0])
            proba = 1.0 if pred >= 0.5 else 0.0
        return proba >= proba_thresh

    return accept
