"""Dagster definitions voor KFL_backend_GPU_v6."""

from dagster import Definitions

from dagster_kfl.assets import (
    kfl_klines_raw,
    kfl_indicators_and_signals,
    kfl_mtf_signals_lead,
    kfl_mtf_signals_coin,
    kfl_mtf_signals_conf,
)
from dagster_kfl.jobs import (
    full_kfl_backfill,
    mtf_only_backfill,
)
from dagster_kfl.resources import BackfillRunConfig


defs = Definitions(
    assets=[
        kfl_klines_raw,
        kfl_indicators_and_signals,
        kfl_mtf_signals_lead,
        kfl_mtf_signals_coin,
        kfl_mtf_signals_conf,
    ],
    jobs=[
        full_kfl_backfill,
        mtf_only_backfill,
    ],
    resources={
        "config": BackfillRunConfig(),
    },
)
