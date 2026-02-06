"""KFL backfill jobs."""

from dagster import AssetSelection, define_asset_job


# Job 1: Volledige KFL backfill (indicators_and_signals + alle MTF)
full_kfl_backfill = define_asset_job(
    name="full_kfl_backfill",
    description="Volledige KFL backfill: indicators, signals, en alle MTF signals",
    selection=AssetSelection.keys(
        ["kfl", "indicators_and_signals"],
        ["kfl", "mtf_signals_lead"],
        ["kfl", "mtf_signals_coin"],
        ["kfl", "mtf_signals_conf"],
    ),
    tags={
        "pipeline": "kfl",
        "compute": "gpu",
    },
)


# Job 2: Alleen MTF backfill (voor snelle MTF updates)
mtf_only_backfill = define_asset_job(
    name="mtf_only_backfill",
    description="Alleen MTF signals backfill (lead, coin, conf)",
    selection=AssetSelection.keys(
        ["kfl", "mtf_signals_lead"],
        ["kfl", "mtf_signals_coin"],
        ["kfl", "mtf_signals_conf"],
    ),
    tags={
        "pipeline": "kfl",
        "mtf_only": "true",
    },
)
