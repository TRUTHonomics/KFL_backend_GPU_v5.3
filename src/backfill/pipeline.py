"""
Main pipeline orchestrator for indicator and signal backfill.

Coordinates all phases:
1. Data loading with lookback buffer
2. Gap detection (optional)
3. GPU indicator calculation (SMA, BB, Stochastic, Ichimoku, OBV, CMF)
4. CPU indicator calculation (RSI, MACD, EMA, DEMA, ATR/ADX, Supertrend, AO)
5. Discrete signal calculation (8 signals)
6. Boolean signal calculation (125 signals)
7. Concordance calculation
8. Database writes via staging tables
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DEFAULT_DB_CONNECTION, LOOKBACK_BUFFER, SIGNAL_COLS
from .data_loader import DataLoader
from .gap_detector import GapDetector
from .indicators.gpu_indicators import GPUIndicatorCalculator
from .indicators.cpu_indicators import CPUIndicatorCalculator
from .indicators.vpvr_indicators import VPVRCalculator
from .signals.discrete_signals import DiscreteSignalCalculator
from .signals.boolean_signals import BooleanSignalCalculator
from .signals.concordance import ConcordanceCalculator
from .writers.staging_manager import StagingManager
from .writers.indicator_writer import IndicatorWriter
from .writers.signal_writer import SignalWriter

logger = logging.getLogger(__name__)


class BackfillPipeline:
    """
    Main orchestrator for indicator and signal backfill.
    
    Modes:
    - 'full': Recalculate everything, overwrite existing data
    - 'gaps_only': Only fill missing timestamps
    - 'incremental': Start from last known timestamp
    """
    
    def __init__(
        self,
        connection_string: str = DEFAULT_DB_CONNECTION,
        use_gpu: bool = True
    ):
        """
        Initialize backfill pipeline.
        
        Args:
            connection_string: PostgreSQL connection string
            use_gpu: Whether to use GPU for vectorized indicators
        """
        self.connection_string = connection_string
        
        # Initialize components
        self.data_loader = DataLoader(connection_string)
        self.gap_detector = GapDetector(connection_string)
        self.gpu_calc = GPUIndicatorCalculator(use_gpu=use_gpu)
        self.cpu_calc = CPUIndicatorCalculator()
        self.vpvr_calc = VPVRCalculator(lookback=50, num_bins=50, value_area_pct=0.70)
        # REASON: Pass connection_string to load thresholds from DB
        self.discrete_calc = DiscreteSignalCalculator(connection_string=connection_string)
        self.boolean_calc = BooleanSignalCalculator(connection_string=connection_string)
        self.concordance_calc = ConcordanceCalculator()
        self.staging_mgr = StagingManager(connection_string)
        self.indicator_writer = IndicatorWriter(connection_string)
        # REASON: Gebruik dezelfde session_id als staging_mgr voor consistente staging tabellen
        self.signal_writer = SignalWriter(connection_string, session_id=self.staging_mgr._session_id)
        
        logger.info(f"BackfillPipeline initialized (session_id={self.staging_mgr._session_id})")
    
    def setup_staging(self, conn=None):
        """Create staging tables for bulk loading."""
        self.staging_mgr.create_indicators_staging(conn)
        
        # Create signals staging tables for each classification
        # REASON: SignalWriter expects these tables to exist before writing
        lead_columns = list(SIGNAL_COLS.LEAD) + list(SIGNAL_COLS.LEAD_DISCRETE) + list(SIGNAL_COLS.CONCORDANCE)
        coin_columns = list(SIGNAL_COLS.COIN) + list(SIGNAL_COLS.COIN_DISCRETE) + list(SIGNAL_COLS.CONCORDANCE)
        conf_columns = list(SIGNAL_COLS.CONF) + list(SIGNAL_COLS.CONF_DISCRETE) + list(SIGNAL_COLS.CONCORDANCE)
        
        self.staging_mgr.create_signals_staging('lead', lead_columns, conn)
        self.staging_mgr.create_signals_staging('coin', coin_columns, conn)
        self.staging_mgr.create_signals_staging('conf', conf_columns, conn)
        
        logger.info("Staging tables created")
    
    def cleanup_staging(self, conn=None):
        """Drop all staging tables."""
        self.staging_mgr.cleanup_all_staging(conn)
    
    def process_asset_interval(
        self,
        asset_id: int,
        interval_min: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        mode_indicators: str = 'full',
        mode_signals: str = 'full',
        conn=None
    ) -> Dict[str, int]:
        """
        Process a single asset+interval combination.
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            start_date: Start date (None = all available)
            end_date: End date (None = now)
            mode_indicators: 'full' or 'gaps_only' for indicators
            mode_signals: 'full' or 'gaps_only' for signals
            conn: Optional existing connection
            
        Returns:
            Dict with rows written per table
        """
        t_start = time.time()
        results = {'indicators': 0, 'signals_lead': 0, 'signals_coin': 0, 'signals_conf': 0}
        
        should_close = conn is None
        if conn is None:
            conn = self.data_loader.get_connection()
        
        try:
            # Determine loading strategy based on indicator mode
            # REASON: Als indicators 'gaps_only' is, laden we alleen nieuwe data.
            # Als signals 'full' is, moeten we alsnog ALLES inladen voor signaal-herberekening.
            # Echter, indicators zijn input voor signalen.
            # Scenario A: Ind=gaps, Sig=gaps -> Laad vanaf gap.
            # Scenario B: Ind=full, Sig=full -> Laad alles.
            # Scenario C: Ind=gaps, Sig=full -> Laad alles, maar schrijf indicators selectief?
            # 
            # In Scenario C (optimalisatie):
            # We willen indicators niet overschrijven (duur), maar wel gebruiken voor signaal berekening.
            # Dus we moeten ALLES laden, herberekenen, en dan slim wegschrijven.
            
            load_full_history = (mode_indicators == 'full') or (mode_signals == 'full')
            
            if not load_full_history:
                # gaps_only mode for BOTH
                first_gap = self.gap_detector.find_first_gap(asset_id, interval_min, end_date, conn)
                
                if first_gap is None:
                    # REASON: Geen echte gap gevonden, maar we forceren 48 uur herberekening 
                    # om instabiele real-time data te corrigeren (indien klines bestaan).
                    from datetime import timedelta, timezone
                    now = datetime.now(timezone.utc)
                    forced_start = (now - timedelta(hours=48))
                    
                    # Check of er überhaupt klines zijn in deze periode
                    exists = self.data_loader.count_klines(asset_id, interval_min, start_date=forced_start, end_date=end_date, conn=conn)
                    if exists == 0:
                        logger.info(f"Asset {asset_id}/{interval_min}: complete, geen nieuwe data in laatste 48u")
                        return results
                        
                    first_gap = forced_start
                    logger.info(f"Asset {asset_id}/{interval_min}: Geen gaps, geforceerde correctie vanaf {first_gap}")
                else:
                    logger.info(f"Asset {asset_id}/{interval_min}: gap gevonden op {first_gap}")
                
                # Load klines starting from gap - lookback
                df, target_start_idx = self.data_loader.load_klines_from_gap(
                    asset_id=asset_id,
                    interval_min=interval_min,
                    gap_time=first_gap,
                    end_date=end_date,
                    conn=conn
                )
                
                if df.empty:
                    logger.warning(f"No klines for asset {asset_id}/{interval_min} from gap")
                    return results
                
                logger.info(f"Loaded {len(df)} klines from gap ({target_start_idx} lookback)")
                
            else:
                # Full mode: Load all klines with lookback buffer
                logger.info(f"Loading klines for asset {asset_id}/{interval_min}")
                df, target_start_idx = self.data_loader.load_klines(
                    asset_id=asset_id,
                    interval_min=interval_min,
                    start_date=start_date,
                    end_date=end_date,
                    include_lookback=True,
                    conn=conn
                )
                
                if df.empty:
                    logger.warning(f"No klines for asset {asset_id}/{interval_min}")
                    return results
                
                logger.info(f"Loaded {len(df)} klines ({target_start_idx} lookback)")
            
            # Convert to numpy
            ohlcv = self.data_loader.to_numpy_arrays(df)
            times = df['time'].values
            
            # Phase 3: GPU indicators
            logger.info("Calculating GPU indicators...")
            t_gpu = time.time()
            gpu_indicators = self.gpu_calc.calculate_all(ohlcv)
            logger.debug(f"GPU indicators: {time.time() - t_gpu:.2f}s")
            
            # Phase 4: CPU indicators
            logger.info("Calculating CPU indicators...")
            t_cpu = time.time()
            cpu_indicators = self.cpu_calc.calculate_all(ohlcv)
            logger.debug(f"CPU indicators: {time.time() - t_cpu:.2f}s")
            
            # Phase 4b: VPVR indicators
            logger.info("Calculating VPVR indicators...")
            t_vpvr = time.time()
            vpvr_indicators = self.vpvr_calc.calculate_all(ohlcv)
            logger.debug(f"VPVR indicators: {time.time() - t_vpvr:.2f}s")
            
            # Merge all indicators
            all_indicators = {**gpu_indicators, **cpu_indicators, **vpvr_indicators}
            
            # Phase 5: Discrete signals
            logger.info("Calculating discrete signals...")
            discrete_signals = self.discrete_calc.calculate_all(ohlcv, all_indicators)
            
            # Phase 6: Boolean signals
            logger.info("Calculating boolean signals...")
            boolean_signals = self.boolean_calc.calculate_all(ohlcv, all_indicators)
            
            # Phase 7: Concordance
            logger.info("Calculating concordance...")
            concordance = self.concordance_calc.calculate_all(boolean_signals, discrete_signals)
            
            # Phase 8+9: Write indicators and signals in parallel
            # REASON: 4 onafhankelijke target tabellen kunnen parallel geschreven worden
            # Elke write krijgt eigen connection (psycopg2 is niet thread-safe)
            logger.info("Writing indicators and signals (parallel)...")
            t_write = time.time()
            
            # Map classification names to table keys
            classification_map = {
                'leading': 'lead',
                'coincident': 'coin',
                'confirming': 'conf',
            }
            
            if mode_indicators == 'gaps_only':
                # OPTIMIZED: Use direct write with DO NOTHING for gap-forward mode
                def write_indicators_task():
                    """Write indicators directly with DO NOTHING"""
                    write_conn = self.data_loader.get_connection()
                    try:
                        return ('indicators', self.indicator_writer.write_indicators_direct(
                            times=times,
                            asset_id=asset_id,
                            interval_min=interval_min,
                            indicators=all_indicators,
                            ohlcv=ohlcv,
                            target_start_idx=target_start_idx,
                            conn=write_conn
                        ))
                    finally:
                        write_conn.close()
            else:
                # Full mode: Use staging table + UPSERT
                write_mask = np.zeros(len(times), dtype=bool)
                write_mask[target_start_idx:] = True
                
                def write_indicators_task():
                    """Write indicators to kfl.indicators"""
                    write_conn = self.data_loader.get_connection()
                    try:
                        return ('indicators', self.indicator_writer.write_indicators(
                            times=times,
                            asset_id=asset_id,
                            interval_min=interval_min,
                            indicators=all_indicators,
                            ohlcv=ohlcv,
                            write_mask=write_mask,
                            conn=write_conn
                        ))
                    finally:
                        write_conn.close()

            # Signal writing logic
            if mode_signals == 'gaps_only':
                def write_signals_task(table_key: str, classification: str):
                    """Write signals directly with DO NOTHING"""
                    write_conn = self.data_loader.get_connection()
                    try:
                        bool_sigs = boolean_signals.get(classification, {})
                        disc_sigs = discrete_signals.get(table_key, {})
                        conc = concordance.get(table_key, {})
                        
                        rows = self.signal_writer.write_signals_direct(
                            table_key=table_key,
                            times=times,
                            asset_id=asset_id,
                            interval_min=interval_min,
                            boolean_signals=bool_sigs,
                            discrete_signals=disc_sigs,
                            concordance=conc,
                            target_start_idx=target_start_idx,
                            conn=write_conn
                        )
                        return (f'signals_{table_key}', rows)
                    finally:
                        write_conn.close()
            else:
                # Full mode: Use staging table + UPSERT
                write_mask = np.zeros(len(times), dtype=bool)
                write_mask[target_start_idx:] = True
                
                def write_signals_task(table_key: str, classification: str):
                    """Write signals to one of the 3 signal tables"""
                    write_conn = self.data_loader.get_connection()
                    try:
                        bool_sigs = boolean_signals.get(classification, {})
                        disc_sigs = discrete_signals.get(table_key, {})
                        conc = concordance.get(table_key, {})
                        
                        rows = self.signal_writer.write_table_signals(
                            table_key=table_key,
                            times=times,
                            asset_id=asset_id,
                            interval_min=interval_min,
                            boolean_signals=bool_sigs,
                            discrete_signals=disc_sigs,
                            concordance=conc,
                            write_mask=write_mask,
                            conn=write_conn
                        )
                        return (f'signals_{table_key}', rows)
                    finally:
                        write_conn.close()
            
            # Execute all 4 writes in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(write_indicators_task),
                    executor.submit(write_signals_task, 'lead', 'leading'),
                    executor.submit(write_signals_task, 'coin', 'coincident'),
                    executor.submit(write_signals_task, 'conf', 'confirming'),
                ]
                
                for future in as_completed(futures):
                    try:
                        key, rows = future.result()
                        results[key] = rows
                    except Exception as e:
                        logger.error(f"Parallel write failed: {e}")
                        raise
            
            logger.debug(f"Parallel writes completed in {time.time() - t_write:.2f}s")
            
            t_total = time.time() - t_start
            logger.info(
                f"Completed asset {asset_id}/{interval_min}: "
                f"{results['indicators']} indicators in {t_total:.2f}s"
            )
            
            return results
            
        finally:
            if should_close:
                conn.close()
    
    def run(
        self,
        asset_ids: Optional[List[int]] = None,
        intervals: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        mode_indicators: str = 'full',
        mode_signals: str = 'full',
        # Backwards compatibility voor single 'mode' argument via **kwargs of logic check?
        # We passen de signatuur aan en updaten de call sites.
        mode: str = 'full' 
    ) -> Dict[str, int]:
        """
        Run the full backfill pipeline.
        
        Args:
            asset_ids: List of asset IDs (None = all)
            intervals: List of interval codes (None = all)
            start_date: Start date (None = all)
            end_date: End date (None = now)
            mode_indicators: Mode for indicators ('full', 'gaps_only')
            mode_signals: Mode for signals ('full', 'gaps_only')
            mode: Legacy fallback (sets both)
            
        Returns:
            Dict with total rows written per table
        """
        # Handle legacy 'mode' argument if specific modes are default
        if mode != 'full' and mode_indicators == 'full' and mode_signals == 'full':
            mode_indicators = mode
            mode_signals = mode
            
        t_start = time.time()
        
        conn = self.data_loader.get_connection()
        
        try:
            # Get assets and intervals if not specified
            if asset_ids is None:
                asset_ids = self.data_loader.get_available_assets(conn)
                logger.info(f"Found {len(asset_ids)} assets")
            
            if intervals is None:
                intervals = self.data_loader.get_available_intervals(conn)
                logger.info(f"Found {len(intervals)} intervals")
            
            # Setup staging tables
            self.setup_staging(conn)
            
            # Track totals
            totals = {'indicators': 0, 'signals_lead': 0, 'signals_coin': 0, 'signals_conf': 0}
            
            # Process each asset+interval
            total_combinations = len(asset_ids) * len(intervals)
            processed = 0
            
            for asset_id in asset_ids:
                for interval_min in intervals:
                    processed += 1
                    logger.info(f"Processing {processed}/{total_combinations}: "
                               f"asset {asset_id}, interval {interval_min}")

                    # Use shared connection - staging tables must be visible
                    try:
                        results = self.process_asset_interval(
                            asset_id=asset_id,
                            interval_min=interval_min,
                            start_date=start_date,
                            end_date=end_date,
                            mode_indicators=mode_indicators,
                            mode_signals=mode_signals,
                            conn=conn  # Use shared connection instead of fresh one
                        )

                        for key in totals:
                            totals[key] += results.get(key, 0)

                        # Commit after successful processing
                        conn.commit()
                        logger.debug(f"Committed asset {asset_id}/{interval_min}")

                    except Exception as e:
                        logger.error(f"Error processing {asset_id}/{interval_min}: {e}")
                        # Rollback to clear aborted transaction and allow next iteration
                        try:
                            conn.rollback()
                            logger.debug("Rolled back transaction after error")
                        except Exception as rb_error:
                            logger.warning(f"Rollback failed: {rb_error}")
            
            # Cleanup staging tables
            self.cleanup_staging(conn)
            
            t_total = time.time() - t_start
            logger.info(
                f"Pipeline completed in {t_total:.2f}s: "
                f"{totals['indicators']} indicators, "
                f"{totals['signals_lead'] + totals['signals_coin'] + totals['signals_conf']} signals"
            )
            
            return totals
            
        finally:
            conn.close()
    
    def run_single(
        self,
        asset_id: int,
        interval_min: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        mode: str = 'full'
    ) -> Dict[str, int]:
        """
        Run pipeline for a single asset+interval (convenience method).
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            start_date: Start date
            end_date: End date
            mode: Processing mode
            
        Returns:
            Dict with rows written
        """
        return self.run(
            asset_ids=[asset_id],
            intervals=[interval_min],
            start_date=start_date,
            end_date=end_date,
            mode=mode
        )
