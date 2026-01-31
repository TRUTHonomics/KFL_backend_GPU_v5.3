"""
KFL Backend GPU v5 - High-Performance Backfill Pipeline

Hybrid GPU/CPU backfill pipeline voor 100M+ klines:
- GPU (CuPy): SMA, BB, Stochastic, Ichimoku, OBV, CMF
- CPU (Numba): RSI, MACD, EMA, DEMA, ATR/ADX, Supertrend, AO
- GPU: All 125 boolean + 8 discrete signals

Schrijft naar historische hypertables (niet naar *_current tabellen).
"""

__version__ = "5.0.0"
