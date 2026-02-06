"""KFL assets package."""

from dagster_kfl.assets.tables import (
    kfl_klines_raw,
    kfl_indicators_and_signals,
    kfl_mtf_signals_lead,
    kfl_mtf_signals_coin,
    kfl_mtf_signals_conf,
)

__all__ = [
    "kfl_klines_raw",
    "kfl_indicators_and_signals",
    "kfl_mtf_signals_lead",
    "kfl_mtf_signals_coin",
    "kfl_mtf_signals_conf",
]
