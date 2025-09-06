from typing import Dict


def fibonacci_retracement(start_price: float, end_price: float) -> Dict[str, float]:
    # Same mapping as production fibo_calculate for consistency
    return {
        "0.0": start_price,
        "0.705": start_price + 0.705 * (end_price - start_price),
        "0.9": start_price + 0.9 * (end_price - start_price),
        "1.0": end_price,
    }


def build_oriented_fib(swing_type: str, impulse_start: float, impulse_end: float) -> Dict[str, float]:
    """
    Orient levels so that for bullish impulse: 0.0 at impulse_end (top), 1.0 at impulse_start (bottom).
    For bearish impulse: 0.0 at impulse_start (top), 1.0 at impulse_end (bottom).
    This matches tests found in test_fib.py.
    """
    if swing_type not in ("bullish", "bearish"):
        raise ValueError("swing_type must be 'bullish' or 'bearish'")

    # Normalize ends
    high = max(impulse_start, impulse_end)
    low = min(impulse_start, impulse_end)
    if swing_type == "bullish":
        # Up impulse: 0.0 at impulse end (higher price), 1.0 at start (lower price)
        raw = fibonacci_retracement(high, low)
    else:
        # Down impulse: 0.0 at impulse start (higher price), 1.0 at end (lower price)
        raw = fibonacci_retracement(low, high)
    return raw
