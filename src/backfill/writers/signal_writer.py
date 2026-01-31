"""
Signal writer using COPY FROM STDIN for maximum performance.

Writes calculated signals to staging tables, then UPSERTs to hypertables.
Uses SignalColumns from config.py to ensure column names match database schema.

CONCURRENCY: Uses session-unique staging table names to prevent conflicts
between concurrent pipeline runs.
"""

import io
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql

from ..config import DEFAULT_DB_CONNECTION, INTERVAL_MINUTES, SIGNAL_COLS

logger = logging.getLogger(__name__)


class SignalWriter:
    """
    High-performance signal writer.
    
    Writes boolean and discrete signals to their respective tables:
    - kfl.signals_lead (leading signals)
    - kfl.signals_coin (coincident signals)
    - kfl.signals_conf (confirming signals)
    
    Uses exact column names from SignalColumns in config.py.
    """
    
    # Column definitions per table from config.py
    LEAD_COLUMNS = list(SIGNAL_COLS.LEAD) + list(SIGNAL_COLS.LEAD_DISCRETE) + list(SIGNAL_COLS.CONCORDANCE)
    COIN_COLUMNS = list(SIGNAL_COLS.COIN) + list(SIGNAL_COLS.COIN_DISCRETE) + list(SIGNAL_COLS.CONCORDANCE)
    CONF_COLUMNS = list(SIGNAL_COLS.CONF) + list(SIGNAL_COLS.CONF_DISCRETE) + list(SIGNAL_COLS.CONCORDANCE)
    
    def __init__(self, connection_string: str = DEFAULT_DB_CONNECTION, session_id: int = None):
        """
        Initialize signal writer.
        
        Args:
            connection_string: PostgreSQL connection string
            session_id: Optional session ID for unique staging table names
        """
        self.connection_string = connection_string
        # REASON: Sessie-unieke staging tabellen voorkomt conflicten bij concurrent runs
        self._session_id = session_id if session_id is not None else os.getpid()
    
    def _get_table_config(self, table_key: str) -> dict:
        """Get table configuration with session-specific staging table name."""
        configs = {
            'lead': {
                'target_table': 'kfl.signals_lead',
                'staging_table': f'staging.signals_lead_backfill_{self._session_id}',
                'columns': self.LEAD_COLUMNS,
            },
            'coin': {
                'target_table': 'kfl.signals_coin',
                'staging_table': f'staging.signals_coin_backfill_{self._session_id}',
                'columns': self.COIN_COLUMNS,
            },
            'conf': {
                'target_table': 'kfl.signals_conf',
                'staging_table': f'staging.signals_conf_backfill_{self._session_id}',
                'columns': self.CONF_COLUMNS,
            },
        }
        return configs[table_key]
    
    def get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(self.connection_string)
    
    def _format_value(self, value) -> str:
        """Format a value for COPY."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return '\\N'
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, (int, np.integer)):
            return str(int(value))
        elif isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return '\\N'
            # Check if it's actually an integer value
            if value == int(value):
                return str(int(value))
            return f"{value:.6f}"
        else:
            return str(value)
    
    def _build_copy_buffer(
        self,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        signals: Dict[str, np.ndarray],
        signal_columns: List[str],
        write_mask: Optional[np.ndarray] = None
    ) -> io.StringIO:
        """
        Build a StringIO buffer for COPY FROM STDIN.
        
        Args:
            times: Array of timestamps
            asset_id: Asset ID
            interval_min: Interval code
            signals: Dict of signal_name -> values array
            signal_columns: Ordered list of signal column names
            write_mask: Optional boolean mask
            
        Returns:
            StringIO buffer ready for COPY
        """
        buffer = io.StringIO()
        
        interval_minutes = INTERVAL_MINUTES.get(interval_min, 1)
        
        n = len(times)
        indices = range(n) if write_mask is None else np.where(write_mask)[0]
        
        # REASON: Converteer times eenmalig vooraf ipv per-rij pd.Timestamp conversie
        if len(times) > 0 and isinstance(times[0], np.datetime64):
            times_py = pd.to_datetime(times).to_pydatetime()
        else:
            times_py = times
        
        # REASON: Bereken time_close vectorized vooraf ipv per-rij timedelta
        delta = timedelta(minutes=interval_minutes)
        times_close = np.array([t + delta for t in times_py])
        
        for i in indices:
            row_values = []
            
            # Time columns (already converted)
            time_dt = times_py[i]
            time_close = times_close[i]
            
            row_values.append(self._format_value(time_dt))
            row_values.append(self._format_value(time_close))
            row_values.append(str(asset_id))
            row_values.append(interval_min)
            
            # Signal columns
            for col in signal_columns:
                if col in signals:
                    row_values.append(self._format_value(signals[col][i]))
                else:
                    row_values.append('\\N')
            
            # REASON: Source tracking voor backfill
            row_values.append('GPU_backfill')
            
            buffer.write('\t'.join(row_values) + '\n')
        
        buffer.seek(0)
        return buffer
    
    def copy_to_staging(
        self,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        signals: Dict[str, np.ndarray],
        signal_columns: List[str],
        staging_table: str,
        write_mask: Optional[np.ndarray] = None,
        conn=None
    ) -> int:
        """
        COPY signals to staging table.
        
        Args:
            times: Array of timestamps
            asset_id: Asset ID
            interval_min: Interval code
            signals: Dict of signal values
            signal_columns: List of signal column names
            staging_table: Target staging table
            write_mask: Optional boolean mask
            conn: Optional existing connection
            
        Returns:
            Number of rows written
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            buffer = self._build_copy_buffer(
                times, asset_id, interval_min, signals, signal_columns, write_mask
            )
            
            row_count = buffer.getvalue().count('\n')
            buffer.seek(0)
            
            if row_count == 0:
                return 0
            
            # REASON: Voeg source_script toe aan de kolommenlijst voor COPY
            all_columns = ['time', 'time_close', 'asset_id', 'interval_min'] + signal_columns + ['source_script']
            
            with conn.cursor() as cur:
                # REASON: TRUNCATE verwijderd hier - upsert_from_staging() doet TRUNCATE na UPSERT
                # Dit voorkomt dubbele TRUNCATE overhead
                schema, table = staging_table.split('.', 1)
                
                # REASON: copy_from kan schema.table niet correct handelen
                # copy_expert geeft volledige controle over de COPY SQL
                columns_str = ', '.join([f'"{c}"' for c in all_columns])
                copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.SQL(columns_str)
                )
                cur.copy_expert(copy_sql, buffer)
                # REASON: Geen commit hier - upsert_from_staging() doet commit na UPSERT
                # Dit combineert COPY + UPSERT in één transactie
            
            logger.debug(f"Copied {row_count} rows to {staging_table}")
            return row_count
            
        finally:
            if should_close:
                conn.close()
    
    def upsert_from_staging(
        self,
        staging_table: str,
        target_table: str,
        signal_columns: List[str],
        conn=None
    ) -> int:
        """
        UPSERT from staging to target table.
        
        Args:
            staging_table: Source staging table
            target_table: Target hypertable
            signal_columns: List of signal column names
            conn: Optional existing connection
            
        Returns:
            Number of rows affected
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # REASON: Voeg source_script toe aan de kolommenlijst
            all_columns = ['time', 'time_close', 'asset_id', 'interval_min'] + signal_columns + ['source_script']
            columns_list = ', '.join([f'"{c}"' for c in all_columns])
            
            # Build UPDATE SET clause - skip PK columns
            update_cols = ['time_close'] + signal_columns + ['source_script']
            update_set = ', '.join([f'"{col}" = EXCLUDED."{col}"' for col in update_cols])
            
            with conn.cursor() as cur:
                # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs
                cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                
                cur.execute(f"""
                    INSERT INTO {target_table} ({columns_list})
                    SELECT {columns_list} FROM {staging_table}
                    ON CONFLICT (asset_id, interval_min, time)
                    DO UPDATE SET {update_set}
                """)
                
                row_count = cur.rowcount
                
                # Clear staging table
                # REASON: Use sql.Identifier to properly quote schema.table name
                schema, table = staging_table.split('.', 1)
                cur.execute(sql.SQL("TRUNCATE TABLE {}.{}").format(
                    sql.Identifier(schema),
                    sql.Identifier(table)
                ))
                conn.commit()
            
            return row_count
            
        finally:
            if should_close:
                conn.close()
    
    def write_table_signals(
        self,
        table_key: str,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        boolean_signals: Dict[str, np.ndarray],
        discrete_signals: Dict[str, np.ndarray],
        concordance: Dict[str, np.ndarray],
        write_mask: Optional[np.ndarray] = None,
        conn=None
    ) -> int:
        """
        Write signals for a specific table.
        
        Args:
            table_key: 'lead', 'coin', or 'conf'
            times: Array of timestamps
            asset_id: Asset ID
            interval_min: Interval code
            boolean_signals: Dict of boolean signal values
            discrete_signals: Dict of discrete signal values
            concordance: Dict with concordance_sum, concordance_count, concordance_score
            write_mask: Optional boolean mask
            conn: Optional existing connection
            
        Returns:
            Number of rows written
        """
        # REASON: Gebruik sessie-specifieke staging tabellen
        config = self._get_table_config(table_key)
        
        # Combine all signals into one dict
        all_signals = {}
        all_signals.update(boolean_signals)
        all_signals.update(discrete_signals)
        all_signals.update(concordance)
        
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # Copy to staging
            rows_copied = self.copy_to_staging(
                times, asset_id, interval_min,
                all_signals, config['columns'],
                config['staging_table'], write_mask, conn
            )
            
            if rows_copied == 0:
                return 0
            
            # Upsert to target
            rows_written = self.upsert_from_staging(
                config['staging_table'],
                config['target_table'],
                config['columns'],
                conn
            )
            
            logger.info(f"Wrote {rows_written} signal rows to {config['target_table']}")
            return rows_written
            
        finally:
            if should_close:
                conn.close()
    
    def write_signals_direct(
        self,
        table_key: str,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        boolean_signals: Dict[str, np.ndarray],
        discrete_signals: Dict[str, np.ndarray],
        concordance: Dict[str, np.ndarray],
        target_start_idx: int,
        conn=None
    ) -> int:
        """
        Write signals directly with INSERT ON CONFLICT DO NOTHING.
        
        Optimized for gap-forward mode: skips staging tables and uses DO NOTHING
        since we know the data doesn't exist yet (from find_first_gap).
        
        Args:
            table_key: 'lead', 'coin', or 'conf'
            times: Array of timestamps
            asset_id: Asset ID
            interval_min: Interval code
            boolean_signals: Dict of boolean signal values
            discrete_signals: Dict of discrete signal values
            concordance: Dict with concordance values
            target_start_idx: Index where new data starts
            conn: Optional existing connection
            
        Returns:
            Number of rows written
        """
        config = self._get_table_config(table_key)
        target_table = config['target_table']
        signal_columns = config['columns']
        
        # Combine all signals into one dict
        all_signals = {}
        all_signals.update(boolean_signals)
        all_signals.update(discrete_signals)
        all_signals.update(concordance)
        
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # Create write_mask from target_start_idx onwards
            write_mask = np.zeros(len(times), dtype=bool)
            write_mask[target_start_idx:] = True
            
            row_count = int(write_mask.sum())
            
            if row_count == 0:
                return 0
            
            # Build buffer for COPY
            buffer = self._build_copy_buffer(
                times, asset_id, interval_min, all_signals, signal_columns, write_mask
            )
            
            # REASON: Voeg source_script toe
            all_columns = ['time', 'time_close', 'asset_id', 'interval_min'] + signal_columns + ['source_script']
            columns_list = ', '.join([f'"{c}"' for c in all_columns])
            
            with conn.cursor() as cur:
                # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs
                cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                
                # Create temp table for this transaction
                temp_table = f'temp_signals_{table_key}_{asset_id}_{interval_min.replace("D", "1440")}'
                
                cur.execute(f"""
                    CREATE TEMP TABLE IF NOT EXISTS {temp_table} (
                        LIKE {target_table} INCLUDING DEFAULTS
                    ) ON COMMIT DROP
                """)
                
                # COPY to temp table using copy_expert
                copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t', NULL '\\N')").format(
                    sql.Identifier(temp_table),
                    sql.SQL(columns_list)
                )
                cur.copy_expert(copy_sql, buffer)
                
                # Build UPDATE SET clause - skip PK columns
                # REASON: Use DO UPDATE om instabiele real-time data te overschrijven
                update_cols = ['time_close'] + signal_columns + ['source_script']
                update_set = ', '.join([f'"{col}" = EXCLUDED."{col}"' for col in update_cols])
                
                # INSERT with DO UPDATE
                cur.execute(f"""
                    INSERT INTO {target_table} ({columns_list})
                    SELECT {columns_list} FROM {temp_table}
                    ON CONFLICT (asset_id, interval_min, time)
                    DO UPDATE SET {update_set}
                """)
                
                rows_inserted = cur.rowcount
                conn.commit()
            
            logger.info(f"Wrote {rows_inserted} signal rows to {target_table} (direct)")
            return rows_inserted
            
        finally:
            if should_close:
                conn.close()
    
    def write_all_signals(
        self,
        times: np.ndarray,
        asset_id: int,
        interval_min: str,
        boolean_signals: Dict[str, Dict[str, np.ndarray]],
        discrete_signals: Dict[str, Dict[str, np.ndarray]],
        concordance: Dict[str, Dict[str, np.ndarray]],
        write_mask: Optional[np.ndarray] = None,
        conn=None
    ) -> Dict[str, int]:
        """
        Write all signals (leading, coincident, confirming).
        
        Args:
            times: Array of timestamps
            asset_id: Asset ID
            interval_min: Interval code
            boolean_signals: Dict with 'leading', 'coincident', 'confirming' keys
            discrete_signals: Dict with 'lead', 'coin', 'conf' keys
            concordance: Dict with 'lead', 'coin', 'conf' keys
            write_mask: Optional boolean mask
            conn: Optional existing connection
            
        Returns:
            Dict mapping table -> rows written
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        results = {}
        
        # Map classification names to table keys
        classification_map = {
            'leading': 'lead',
            'coincident': 'coin',
            'confirming': 'conf',
        }
        
        try:
            for classification, table_key in classification_map.items():
                bool_sigs = boolean_signals.get(classification, {})
                disc_sigs = discrete_signals.get(table_key, {})
                conc = concordance.get(table_key, {})
                
                rows = self.write_table_signals(
                    table_key, times, asset_id, interval_min,
                    bool_sigs, disc_sigs, conc,
                    write_mask, conn
                )
                
                results[self._get_table_config(table_key)['target_table']] = rows
            
            return results
            
        finally:
            if should_close:
                conn.close()
