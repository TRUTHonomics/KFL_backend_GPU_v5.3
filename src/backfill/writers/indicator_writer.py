"""
Indicator writer using COPY FROM STDIN for maximum performance.

Writes calculated indicators to staging tables, then UPSERTs to hypertables.
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql

from ..config import DEFAULT_DB_CONNECTION, INTERVAL_MINUTES

logger = logging.getLogger(__name__)


class IndicatorWriter:
    """
    High-performance indicator writer.
    
    Uses PostgreSQL COPY protocol for bulk inserts to staging,
    then UPSERTs to final hypertables.
    """
    
    # Columns matching kfl.indicators table schema
    # REASON: Must match actual database columns exactly
    INDICATOR_COLUMNS = [
        'time', 'asset_id', 'interval_min',
        # OHLCV (required by table)
        'open', 'high', 'low', 'close', 'volume',
        # RSI
        'rsi_7', 'rsi_14', 'rsi_21',
        # MACD variants (db uses different naming)
        'macd_12_26_9', 'macd_12_26_9_signal', 'macd_12_26_9_histogram',
        'macd_6_13_4', 'macd_6_13_4_signal', 'macd_6_13_4_histogram',
        'macd_20_50_15', 'macd_20_50_15_signal', 'macd_20_50_15_histogram',
        'macd_8_24_9', 'macd_8_24_9_signal', 'macd_8_24_9_histogram',
        'macd_5_35_5', 'macd_5_35_5_signal', 'macd_5_35_5_histogram',
        # SMA
        'sma_20', 'sma_50', 'sma_100', 'sma_200',
        # EMA
        'ema_10', 'ema_12', 'ema_20', 'ema_26', 'ema_50', 'ema_100', 'ema_200',
        # DEMA
        'dema_10', 'dema_20', 'dema_50', 'dema_100', 'dema_200',
        # Ichimoku (db naming without ichi_ prefix)
        'tenkan_9_26_52', 'kijun_9_26_52', 'senkou_a_9_26_52', 'senkou_b_9_26_52', 'chikou_9_26_52',
        'tenkan_10_30_60', 'kijun_10_30_60', 'senkou_a_10_30_60', 'senkou_b_10_30_60', 'chikou_10_30_60',
        'tenkan_7_22_44', 'kijun_7_22_44', 'senkou_a_7_22_44', 'senkou_b_7_22_44', 'chikou_7_22_44',
        'tenkan_6_20_52', 'kijun_6_20_52', 'senkou_a_6_20_52', 'senkou_b_6_20_52', 'chikou_6_20_52',
        # ATR/ADX
        'atr_14', 'atr_10', 'atr_ma', 'adx_14', 'dm_plus_14', 'dm_minus_14',
        # Stochastic
        'stoch_k', 'stoch_d',
        # Bollinger Bands
        'bb_upper', 'bb_middle', 'bb_lower',
        # CMF
        'cmf',
        # Keltner
        'keltner_upper', 'keltner_middle', 'keltner_lower',
        # OBV
        'obv',
        # AO
        'ao_5_34',
        # Supertrend (only one variant in db)
        'supertrend_10_3', 'supertrend_direction',
        # VPVR (Volume Profile Visible Range)
        'vpvr_poc', 'vpvr_vah', 'vpvr_val',
        'vpvr_hvn_upper', 'vpvr_hvn_lower',
        'vpvr_lvn_upper', 'vpvr_lvn_lower',
        # REASON: Source tracking voor backfill
        'source_script',
    ]
    
    # Mapping from calculated indicator names to database column names
    COLUMN_MAPPING = {
        # RSI - direct match
        'rsi_7': 'rsi_7',
        'rsi_14': 'rsi_14',
        'rsi_21': 'rsi_21',
        # MACD - direct naming from calculator results
        'macd_12_26_9': 'macd_12_26_9',
        'macd_12_26_9_signal': 'macd_12_26_9_signal',
        'macd_12_26_9_histogram': 'macd_12_26_9_histogram',
        'macd_6_13_4': 'macd_6_13_4',
        'macd_6_13_4_signal': 'macd_6_13_4_signal',
        'macd_6_13_4_histogram': 'macd_6_13_4_histogram',
        'macd_20_50_15': 'macd_20_50_15',
        'macd_20_50_15_signal': 'macd_20_50_15_signal',
        'macd_20_50_15_histogram': 'macd_20_50_15_histogram',
        'macd_8_24_9': 'macd_8_24_9',
        'macd_8_24_9_signal': 'macd_8_24_9_signal',
        'macd_8_24_9_histogram': 'macd_8_24_9_histogram',
        'macd_5_35_5': 'macd_5_35_5',
        'macd_5_35_5_signal': 'macd_5_35_5_signal',
        'macd_5_35_5_histogram': 'macd_5_35_5_histogram',
        # SMA
        'sma_20': 'sma_20',
        'sma_50': 'sma_50',
        'sma_100': 'sma_100',
        'sma_200': 'sma_200',
        # EMA
        'ema_10': 'ema_10',
        'ema_12': 'ema_12',
        'ema_20': 'ema_20',
        'ema_26': 'ema_26',
        'ema_50': 'ema_50',
        'ema_100': 'ema_100',
        'ema_200': 'ema_200',
        # DEMA
        'dema_10': 'dema_10',
        'dema_20': 'dema_20',
        'dema_50': 'dema_50',
        'dema_100': 'dema_100',
        'dema_200': 'dema_200',
        # Ichimoku - remove ichi_ prefix
        'ichi_tenkan_9_26_52': 'tenkan_9_26_52',
        'ichi_kijun_9_26_52': 'kijun_9_26_52',
        'ichi_senkou_a_9_26_52': 'senkou_a_9_26_52',
        'ichi_senkou_b_9_26_52': 'senkou_b_9_26_52',
        'ichi_chikou_9_26_52': 'chikou_9_26_52',
        'ichi_tenkan_10_30_60': 'tenkan_10_30_60',
        'ichi_kijun_10_30_60': 'kijun_10_30_60',
        'ichi_senkou_a_10_30_60': 'senkou_a_10_30_60',
        'ichi_senkou_b_10_30_60': 'senkou_b_10_30_60',
        'ichi_chikou_10_30_60': 'chikou_10_30_60',
        'ichi_tenkan_7_22_44': 'tenkan_7_22_44',
        'ichi_kijun_7_22_44': 'kijun_7_22_44',
        'ichi_senkou_a_7_22_44': 'senkou_a_7_22_44',
        'ichi_senkou_b_7_22_44': 'senkou_b_7_22_44',
        'ichi_chikou_7_22_44': 'chikou_7_22_44',
        'ichi_tenkan_6_20_52': 'tenkan_6_20_52',
        'ichi_kijun_6_20_52': 'kijun_6_20_52',
        'ichi_senkou_a_6_20_52': 'senkou_a_6_20_52',
        'ichi_senkou_b_6_20_52': 'senkou_b_6_20_52',
        'ichi_chikou_6_20_52': 'chikou_6_20_52',
        # ATR/ADX
        'atr_14': 'atr_14',
        'atr_10': 'atr_10',
        'atr_ma': 'atr_ma',
        'adx_14': 'adx_14',
        'plus_di_14': 'dm_plus_14',
        'minus_di_14': 'dm_minus_14',
        # Direct matches
        'stoch_k': 'stoch_k',
        'stoch_d': 'stoch_d',
        'bb_upper': 'bb_upper',
        'bb_middle': 'bb_middle',
        'bb_lower': 'bb_lower',
        'cmf': 'cmf',
        'keltner_middle': 'keltner_middle',
        'keltner_upper': 'keltner_upper',
        'keltner_lower': 'keltner_lower',
        'obv': 'obv',
        'ao': 'ao_5_34',
        'supertrend_10_3': 'supertrend_10_3',
        'supertrend_dir_10_3': 'supertrend_direction',
        # VPVR - direct match (names match database columns)
        'vpvr_poc': 'vpvr_poc',
        'vpvr_vah': 'vpvr_vah',
        'vpvr_val': 'vpvr_val',
        'vpvr_hvn_upper': 'vpvr_hvn_upper',
        'vpvr_hvn_lower': 'vpvr_hvn_lower',
        'vpvr_lvn_upper': 'vpvr_lvn_upper',
        'vpvr_lvn_lower': 'vpvr_lvn_lower',
    }
    
    def __init__(self, connection_string: str = DEFAULT_DB_CONNECTION):
        """
        Initialize indicator writer.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        # REASON: Precompute inverse mapping voor O(1) lookup ipv O(n) per cel
        self._db_to_calc = {}
        for calc, db in self.COLUMN_MAPPING.items():
            self._db_to_calc.setdefault(db, calc)
    
    def get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(self.connection_string)
    
    def _format_value(self, value) -> str:
        """Format a value for COPY."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return '\\N'  # NULL in COPY format
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (int, np.integer)):
            return str(int(value))
        elif isinstance(value, (float, np.floating)):
            return f'{float(value):.10g}'  # Avoid scientific notation for small values
        else:
            return str(value)
    
    def _build_copy_buffer(
        self,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        indicators: Dict[str, np.ndarray],
        ohlcv: Dict[str, np.ndarray] = None,
        write_mask: Optional[np.ndarray] = None
    ) -> io.StringIO:
        """
        Build a StringIO buffer for COPY FROM STDIN.
        
        Args:
            times: Array of timestamps (datetime64 or datetime)
            asset_id: Asset ID
            interval_min: Interval code
            indicators: Dict of indicator_name -> values array
            ohlcv: Dict with open, high, low, close, volume arrays
            write_mask: Optional boolean mask for which rows to write
            
        Returns:
            StringIO buffer ready for COPY
        """
        buffer = io.StringIO()
        
        n = len(times)
        indices = range(n) if write_mask is None else np.where(write_mask)[0]
        
        # REASON: Converteer times eenmalig vooraf ipv per-rij pd.Timestamp conversie
        if len(times) > 0 and isinstance(times[0], np.datetime64):
            times_py = pd.to_datetime(times).to_pydatetime()
        else:
            times_py = times
        
        for i in indices:
            row_values = []
            
            # Time (already converted)
            time_dt = times_py[i]
            row_values.append(self._format_value(time_dt))
            row_values.append(str(asset_id))
            row_values.append(interval_min)
            
            # OHLCV columns (after time, asset_id, interval_min)
            if ohlcv:
                row_values.append(self._format_value(ohlcv['open'][i]))
                row_values.append(self._format_value(ohlcv['high'][i]))
                row_values.append(self._format_value(ohlcv['low'][i]))
                row_values.append(self._format_value(ohlcv['close'][i]))
                row_values.append(self._format_value(ohlcv['volume'][i]))
            else:
                # No OHLCV provided - use NULL
                row_values.extend(['\\N'] * 5)
            
            # Indicator columns (skip time, asset_id, interval_min, OHLCV)
            for col in self.INDICATOR_COLUMNS[8:]:  # Skip first 8 columns
                # Check if we have this column directly
                if col in indicators:
                    row_values.append(self._format_value(indicators[col][i]))
                else:
                    # REASON: O(1) lookup via precomputed inverse mapping ipv O(n) loop
                    calc_name = self._db_to_calc.get(col)
                    if calc_name and calc_name in indicators:
                        row_values.append(self._format_value(indicators[calc_name][i]))
                    elif col == 'source_script':
                        # REASON: Hardcoded waarde voor backfill tracking
                        row_values.append('GPU_backfill')
                    else:
                        row_values.append('\\N')  # NULL for missing
            
            buffer.write('\t'.join(row_values) + '\n')
        
        buffer.seek(0)
        return buffer
    
    def copy_to_staging(
        self,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        indicators: Dict[str, np.ndarray],
        write_mask: Optional[np.ndarray] = None,
        staging_table: str = 'staging.indicators_backfill',
        conn=None
    ) -> int:
        """
        COPY indicators to staging table.
        
        Args:
            times: Array of timestamps
            asset_id: Asset ID
            interval_min: Interval code
            indicators: Dict of indicator values
            write_mask: Optional boolean mask for which rows to write
            staging_table: Target staging table
            conn: Optional existing connection
            
        Returns:
            Number of rows written
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # REASON: Bereken row_count direct ipv buffer.getvalue().count('\n') (vermijdt string copy)
            row_count = len(times) if write_mask is None else int(write_mask.sum())
            
            if row_count == 0:
                logger.debug(f"No rows to write for asset {asset_id}/{interval_min}")
                return 0
            
            buffer = self._build_copy_buffer(
                times, asset_id, interval_min, indicators, write_mask
            )
            
            with conn.cursor() as cur:
                cur.copy_from(
                    buffer,
                    staging_table,
                    sep='\t',
                    null='\\N',
                    columns=self.INDICATOR_COLUMNS
                )
                conn.commit()
            
            logger.debug(f"Copied {row_count} rows to {staging_table}")
            return row_count
            
        finally:
            if should_close:
                conn.close()
    
    def write_indicators(
        self,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        indicators: Dict[str, np.ndarray],
        ohlcv: Dict[str, np.ndarray] = None,
        write_mask: Optional[np.ndarray] = None,
        staging_table: str = 'staging.indicators_backfill',
        target_table: str = 'kfl.indicators',
        conn=None
    ) -> int:
        """
        Write indicators directly to target table using COPY + temp table.
        
        This is the main entry point for writing indicators.
        
        Args:
            times: Array of timestamps
            asset_id: Asset ID
            interval_min: Interval code
            indicators: Dict of indicator values
            ohlcv: Dict with open, high, low, close, volume arrays
            write_mask: Optional boolean mask
            staging_table: Staging table name (unused, kept for API compatibility)
            target_table: Target hypertable
            conn: Optional existing connection
            
        Returns:
            Number of rows written
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # REASON: Bereken row_count direct ipv buffer.getvalue().count('\n') (vermijdt string copy)
            row_count = len(times) if write_mask is None else int(write_mask.sum())
            
            if row_count == 0:
                logger.debug(f"No rows to write for asset {asset_id}/{interval_min}")
                return 0
            
            # Build buffer for COPY
            buffer = self._build_copy_buffer(
                times, asset_id, interval_min, indicators, ohlcv, write_mask
            )
            
            with conn.cursor() as cur:
                # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs
                cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                
                # Create temp table for this transaction
                temp_table = f'temp_indicators_{asset_id}_{interval_min.replace("D", "1440")}'
                
                # Use a temp table that auto-drops at end of transaction
                cur.execute(f"""
                    CREATE TEMP TABLE IF NOT EXISTS {temp_table} (
                        LIKE {target_table} INCLUDING DEFAULTS
                    ) ON COMMIT DROP
                """)
                
                # COPY to temp table
                cur.copy_from(
                    buffer,
                    temp_table,
                    sep='\t',
                    null='\\N',
                    columns=self.INDICATOR_COLUMNS
                )
                
                # UPSERT to target
                columns_list = ', '.join(self.INDICATOR_COLUMNS)
                update_set = ', '.join([
                    f'{col} = EXCLUDED.{col}' 
                    for col in self.INDICATOR_COLUMNS 
                    if col not in ('time', 'asset_id', 'interval_min')
                ])
                
                cur.execute(f"""
                    INSERT INTO {target_table} ({columns_list})
                    SELECT {columns_list} FROM {temp_table}
                    ON CONFLICT (time, asset_id, interval_min)
                    DO UPDATE SET {update_set}
                """)
                
                conn.commit()
            
            logger.info(f"Wrote {row_count} indicator rows for asset {asset_id}/{interval_min}")
            return row_count
            
        finally:
            if should_close:
                conn.close()
    
    def write_indicators_direct(
        self,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        indicators: Dict[str, np.ndarray],
        ohlcv: Dict[str, np.ndarray],
        target_start_idx: int,
        target_table: str = 'kfl.indicators',
        conn=None
    ) -> int:
        """
        Write indicators directly with INSERT ON CONFLICT DO NOTHING.
        
        Optimized for gap-forward mode: skips staging tables and uses DO NOTHING
        since we know the data doesn't exist yet (from find_first_gap).
        
        Args:
            times: Array of timestamps
            asset_id: Asset ID
            interval_min: Interval code
            indicators: Dict of indicator values
            ohlcv: Dict with open, high, low, close, volume arrays
            target_start_idx: Index where new data starts (rows before this are lookback)
            target_table: Target hypertable
            conn: Optional existing connection
            
        Returns:
            Number of rows written
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # Create write_mask from target_start_idx onwards
            write_mask = np.zeros(len(times), dtype=bool)
            write_mask[target_start_idx:] = True
            
            row_count = int(write_mask.sum())
            
            if row_count == 0:
                logger.debug(f"No rows to write for asset {asset_id}/{interval_min}")
                return 0
            
            # Build buffer for COPY (only rows from target_start_idx onwards)
            buffer = self._build_copy_buffer(
                times, asset_id, interval_min, indicators, ohlcv, write_mask
            )
            
            with conn.cursor() as cur:
                # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs
                cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                
                # Create temp table for this transaction
                temp_table = f'temp_indicators_direct_{asset_id}_{interval_min.replace("D", "1440")}'
                
                cur.execute(f"""
                    CREATE TEMP TABLE IF NOT EXISTS {temp_table} (
                        LIKE {target_table} INCLUDING DEFAULTS
                    ) ON COMMIT DROP
                """)
                
                # COPY to temp table
                cur.copy_from(
                    buffer,
                    temp_table,
                    sep='\t',
                    null='\\N',
                    columns=self.INDICATOR_COLUMNS
                )
                
                # UPSERT to target
                # REASON: Use DO UPDATE to overwrite potentially unstable real-time values
                columns_list = ', '.join(self.INDICATOR_COLUMNS)
                update_set = ', '.join([
                    f'{col} = EXCLUDED.{col}' 
                    for col in self.INDICATOR_COLUMNS 
                    if col not in ('time', 'asset_id', 'interval_min')
                ])
                
                cur.execute(f"""
                    INSERT INTO {target_table} ({columns_list})
                    SELECT {columns_list} FROM {temp_table}
                    ON CONFLICT (time, asset_id, interval_min)
                    DO UPDATE SET {update_set}
                """)
                
                rows_inserted = cur.rowcount
                conn.commit()
            
            logger.info(f"Wrote {rows_inserted} indicator rows for asset {asset_id}/{interval_min} (direct)")
            return rows_inserted
            
        finally:
            if should_close:
                conn.close()
    
    def write_batch(
        self,
        batch_data: List[Dict],
        staging_table: str = 'staging.indicators_backfill',
        target_table: str = 'kfl.indicators',
        conn=None
    ) -> int:
        """
        Write multiple asset/interval combinations in a single transaction.
        
        Args:
            batch_data: List of dicts with keys:
                - times: np.ndarray
                - asset_id: int
                - interval_min: str
                - indicators: Dict[str, np.ndarray]
                - write_mask: Optional[np.ndarray]
            staging_table: Staging table name
            target_table: Target hypertable
            conn: Optional existing connection
            
        Returns:
            Total number of rows written
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        total_rows = 0
        
        try:
            # REASON: Bereken row_count direct ipv buffer.getvalue().count('\n') (vermijdt string copy)
            for item in batch_data:
                wm = item.get('write_mask')
                total_rows += len(item['times']) if wm is None else int(wm.sum())
            
            if total_rows == 0:
                return 0
            
            # Build combined buffer
            combined_buffer = io.StringIO()
            
            for item in batch_data:
                buffer = self._build_copy_buffer(
                    item['times'],
                    item['asset_id'],
                    item['interval_min'],
                    item['indicators'],
                    item.get('write_mask')
                )
                combined_buffer.write(buffer.read())
            
            combined_buffer.seek(0)
            
            # COPY all at once
            with conn.cursor() as cur:
                # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs
                cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                
                cur.copy_from(
                    combined_buffer,
                    staging_table,
                    sep='\t',
                    null='\\N',
                    columns=self.INDICATOR_COLUMNS
                )
                
                # UPSERT to target
                columns_list = ', '.join(self.INDICATOR_COLUMNS)
                update_set = ', '.join([
                    f'{col} = EXCLUDED.{col}' 
                    for col in self.INDICATOR_COLUMNS 
                    if col not in ('time', 'asset_id', 'interval_min')
                ])
                
                cur.execute(f"""
                    INSERT INTO {target_table} ({columns_list})
                    SELECT {columns_list} FROM {staging_table}
                    ON CONFLICT (time, asset_id, interval_min)
                    DO UPDATE SET {update_set}
                """)
                
                # Clear staging
                cur.execute(f"TRUNCATE TABLE {staging_table}")
                conn.commit()
            
            logger.info(f"Wrote batch of {total_rows} indicator rows")
            return total_rows
            
        finally:
            if should_close:
                conn.close()
