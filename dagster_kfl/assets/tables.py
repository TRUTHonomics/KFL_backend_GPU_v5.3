"""KFL table assets: klines_raw, indicators_and_signals, MTF signals."""

import subprocess
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    MetadataValue,
    SourceAsset,
    asset,
    AssetKey,
)

from dagster_kfl.resources.backfill_config import BackfillRunConfig


# REASON: klines_raw wordt extern geproduceerd door real-time pipeline op VM104
kfl_klines_raw = SourceAsset(
    key=AssetKey(["kfl", "klines_raw"]),
    description="Ruwe OHLCV kline data geproduceerd door real-time WebSocket pipeline (VM104)",
)


def _run_kfl_script(
    script_or_module: str,
    context: AssetExecutionContext,
    extra_args: list[str] | None = None,
    timeout: int = 7200,
) -> dict:
    """Voer een Python script uit in de KFL_backend_GPU_v6 container.
    
    Args:
        script_or_module: Module path (bijv. 'backfill.cli')
        context: Dagster context voor logging
        extra_args: Extra command-line argumenten
        timeout: Timeout in seconden (standaard 2 uur)
    
    Returns:
        Dict met stdout, stderr, en returncode
    """
    cmd = ["docker", "exec", "KFL_backend_GPU_v6", "python", "-m", script_or_module]
    if extra_args:
        cmd.extend(extra_args)
    
    context.log.info(f"Executing: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    
    if result.stdout:
        context.log.info(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        context.log.warning(f"STDERR:\n{result.stderr}")
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed with exit code {result.returncode}\n"
            f"STDERR: {result.stderr}\n"
            f"STDOUT: {result.stdout}"
        )
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@asset(
    key=AssetKey(["kfl", "indicators_and_signals"]),
    deps=[kfl_klines_raw],
    description="GPU-accelerated backfill: berekent indicators + signals (lead/coin/conf) via backfill.cli",
    group_name="kfl_backfill",
    compute_kind="docker",
)
def kfl_indicators_and_signals(
    context: AssetExecutionContext,
    config: BackfillRunConfig,
) -> None:
    """Voert volledige KFL backfill uit (indicators + signals).
    
    REASON: De backfill.cli voert indicators + signals in een run uit.
    We modelleren dit als een enkele asset omdat ze niet los uitvoerbaar zijn.
    """
    args = [
        "--asset_id", str(config.asset_id),
        "--mode", config.mode,
        "--intervals", config.intervals,
    ]
    
    if config.start_date:
        args.extend(["--start", config.start_date])
    if config.end_date:
        args.extend(["--end", config.end_date])
    if not config.use_gpu:
        args.append("--no-gpu")
    
    result = _run_kfl_script("backfill.cli", context, extra_args=args)
    
    # Parse output voor metadata (optioneel)
    context.log.info(
        "Backfill voltooid",
        metadata={
            "asset_id": config.asset_id,
            "mode": config.mode,
            "intervals": config.intervals,
            "gpu_enabled": config.use_gpu,
            "stdout_lines": len(result["stdout"].split("\n")),
        },
    )


@asset(
    key=AssetKey(["kfl", "mtf_signals_lead"]),
    deps=[AssetKey(["kfl", "indicators_and_signals"])],
    description="Multi-TimeFrame signals voor lead indicator",
    group_name="kfl_mtf",
    compute_kind="docker",
)
def kfl_mtf_signals_lead(
    context: AssetExecutionContext,
    config: BackfillRunConfig,
) -> None:
    """Genereer MTF signals voor lead indicator."""
    # REASON: MTF backfill wordt via aparte module uitgevoerd
    args = [
        "--asset_id", str(config.asset_id),
        "--intervals", config.intervals,
        "--signal-type", "lead",
    ]
    
    result = _run_kfl_script("backfill.mtf_writer", context, extra_args=args)
    
    context.log.info(
        "MTF lead signals voltooid",
        metadata={
            "asset_id": config.asset_id,
            "intervals": config.intervals,
        },
    )


@asset(
    key=AssetKey(["kfl", "mtf_signals_coin"]),
    deps=[AssetKey(["kfl", "indicators_and_signals"])],
    description="Multi-TimeFrame signals voor coin indicator",
    group_name="kfl_mtf",
    compute_kind="docker",
)
def kfl_mtf_signals_coin(
    context: AssetExecutionContext,
    config: BackfillRunConfig,
) -> None:
    """Genereer MTF signals voor coin indicator."""
    args = [
        "--asset_id", str(config.asset_id),
        "--intervals", config.intervals,
        "--signal-type", "coin",
    ]
    
    result = _run_kfl_script("backfill.mtf_writer", context, extra_args=args)
    
    context.log.info(
        "MTF coin signals voltooid",
        metadata={
            "asset_id": config.asset_id,
            "intervals": config.intervals,
        },
    )


@asset(
    key=AssetKey(["kfl", "mtf_signals_conf"]),
    deps=[AssetKey(["kfl", "indicators_and_signals"])],
    description="Multi-TimeFrame signals voor conf indicator",
    group_name="kfl_mtf",
    compute_kind="docker",
)
def kfl_mtf_signals_conf(
    context: AssetExecutionContext,
    config: BackfillRunConfig,
) -> None:
    """Genereer MTF signals voor conf indicator."""
    args = [
        "--asset_id", str(config.asset_id),
        "--intervals", config.intervals,
        "--signal-type", "conf",
    ]
    
    result = _run_kfl_script("backfill.mtf_writer", context, extra_args=args)
    
    context.log.info(
        "MTF conf signals voltooid",
        metadata={
            "asset_id": config.asset_id,
            "intervals": config.intervals,
        },
    )
