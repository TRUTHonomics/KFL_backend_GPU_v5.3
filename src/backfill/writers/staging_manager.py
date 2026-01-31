"""
Staging table management for bulk data loading.

Creates temporary staging tables for efficient COPY FROM STDIN,
then UPSERTs to final hypertables.

CONCURRENCY: Uses session-unique table names (based on pg_backend_pid)
to prevent conflicts between concurrent pipeline runs.
"""

import logging
import os
from typing import List, Optional

import psycopg2
from psycopg2 import sql

from ..config import DEFAULT_DB_CONNECTION

logger = logging.getLogger(__name__)


class StagingManager:
    """
    Manages staging tables for efficient bulk loading.
    
    Strategy:
    1. Create unlogged staging tables (no WAL overhead)
    2. COPY FROM STDIN to staging (fastest insert method)
    3. UPSERT from staging to hypertable
    4. Drop/truncate staging tables
    """
    
    def __init__(self, connection_string: str = DEFAULT_DB_CONNECTION):
        """
        Initialize staging manager.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        # REASON: Gebruik proces ID voor sessie-unieke staging tabellen
        # Dit voorkomt conflicten tussen concurrent pipeline runs
        self._session_id = os.getpid()
        self._staging_tables = []  # Track aangemaakte tabellen voor cleanup
    
    def get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(self.connection_string)
    
    def create_indicators_staging(self, conn=None) -> str:
        """
        Create staging table for indicators.
        
        Returns:
            Name of the staging table
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        # REASON: Sessie-unieke tabelnaam voorkomt conflicten bij concurrent runs
        staging_table = f'staging.indicators_backfill_{self._session_id}'
        
        # REASON: Gebruik LIKE kfl.indicators om de exacte schema van de doeltabel te volgen
        # Dit is robuuster tegen schema wijzigingen (zoals de toevoeging van source_script)
        try:
            with conn.cursor() as cur:
                # Ensure staging schema exists
                cur.execute("CREATE SCHEMA IF NOT EXISTS staging")
                
                # Drop if exists
                cur.execute(f"DROP TABLE IF EXISTS {staging_table}")
                
                # Create unlogged table (faster, no WAL)
                cur.execute(f"""
                    CREATE UNLOGGED TABLE {staging_table} (
                        LIKE kfl.indicators INCLUDING DEFAULTS
                    )
                """)
                
                # REASON: kfl.indicators heeft created_at/updated_at defaults, 
                # maar we willen tijdstippen van backfill behouden. 
                # Staging tabel heeft geen extra aanpassingen nodig.
                
                conn.commit()
                self._staging_tables.append(staging_table)
                logger.info(f"Created staging table: {staging_table}")
                
        finally:
            if should_close:
                conn.close()
        
        return staging_table
    
    def get_indicators_staging_table(self) -> str:
        """Return the session-specific indicators staging table name."""
        return f'staging.indicators_backfill_{self._session_id}'
    
    def create_signals_staging(
        self,
        classification: str,
        signal_columns: List[str],
        conn=None
    ) -> str:
        """
        Create staging table for signals.
        
        Args:
            classification: 'lead', 'coin', or 'conf'
            signal_columns: List of signal column names
            conn: Optional existing connection
            
        Returns:
            Name of the staging table
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        # REASON: Sessie-unieke tabelnaam voorkomt conflicten bij concurrent runs
        staging_table = f'staging.signals_{classification}_backfill_{self._session_id}'
        
        try:
            with conn.cursor() as cur:
                # Ensure staging schema exists
                cur.execute("CREATE SCHEMA IF NOT EXISTS staging")
                
                # Drop if exists
                cur.execute(f"DROP TABLE IF EXISTS {staging_table}")
                
                # Build column definitions
                # REASON: concordance kolommen hebben specifieke types
                def get_col_type(col: str) -> str:
                    if col == 'concordance_score':
                        # REASON: DOUBLE PRECISION ipv NUMERIC voor betere bulk load performance
                        return 'DOUBLE PRECISION'
                    elif col in ('concordance_sum', 'concordance_count'):
                        return 'INTEGER'
                    else:
                        return 'SMALLINT'
                
                signal_cols_def = ', '.join([
                    f'{col} {get_col_type(col)}' for col in signal_columns
                ])
                
                # Create unlogged table
                cur.execute(f"""
                    CREATE UNLOGGED TABLE {staging_table} (
                        time TIMESTAMPTZ NOT NULL,
                        time_close TIMESTAMPTZ NOT NULL,
                        asset_id INTEGER NOT NULL,
                        interval_min kfl.interval_type NOT NULL,
                        {signal_cols_def},
                        source_script VARCHAR(50)
                    )
                """)
                
                conn.commit()
                self._staging_tables.append(staging_table)
                # DEBUG: Verify table exists after creation
                cur.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='staging' AND table_name LIKE '%backfill_{self._session_id}'")
                count = cur.fetchone()[0]
                logger.info(f"Created staging table: {staging_table} with {len(signal_columns)} signals [VERIFY: {count} tables with suffix _{self._session_id} exist]")
                
        finally:
            if should_close:
                conn.close()
        
        return staging_table
    
    def get_signals_staging_table(self, classification: str) -> str:
        """Return the session-specific signals staging table name."""
        return f'staging.signals_{classification}_backfill_{self._session_id}'
    
    def truncate_staging(self, staging_table: str, conn=None):
        """
        Truncate a staging table (faster than DELETE).
        
        Args:
            staging_table: Full table name (schema.table)
            conn: Optional existing connection
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {staging_table}")
                conn.commit()
                logger.debug(f"Truncated {staging_table}")
        finally:
            if should_close:
                conn.close()
    
    def drop_staging(self, staging_table: str, conn=None):
        """
        Drop a staging table.
        
        Args:
            staging_table: Full table name (schema.table)
            conn: Optional existing connection
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {staging_table}")
                conn.commit()
                logger.info(f"Dropped staging table: {staging_table}")
        finally:
            if should_close:
                conn.close()
    
    def upsert_indicators(
        self,
        staging_table: str = 'staging.indicators_backfill',
        target_table: str = 'kfl.indicators',
        conn=None
    ) -> int:
        """
        UPSERT from staging to target indicators table.
        
        Args:
            staging_table: Source staging table
            target_table: Target hypertable
            conn: Optional existing connection
            
        Returns:
            Number of rows affected
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs
                cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                
                # REASON: Gebruik sessie-specifieke tabelnaam ipv hardcoded 'indicators_backfill'
                # Dit is zowel correct (tabelnaam bevat _pid suffix) als efficiënter
                schema_name, table_name = staging_table.split('.', 1)
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = %s 
                    AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema_name, table_name))
                all_columns = [row[0] for row in cur.fetchall()]
                
                # Build UPSERT query
                columns_list = ', '.join(all_columns)
                update_set = ', '.join([
                    f'{col} = EXCLUDED.{col}' 
                    for col in all_columns 
                    if col not in ('time', 'asset_id', 'interval_min')
                ])
                
                upsert_query = f"""
                    INSERT INTO {target_table} ({columns_list})
                    SELECT {columns_list} FROM {staging_table}
                    ON CONFLICT (time, asset_id, interval_min)
                    DO UPDATE SET {update_set}
                """
                
                cur.execute(upsert_query)
                rows_affected = cur.rowcount
                conn.commit()
                
                logger.info(f"Upserted {rows_affected} rows from {staging_table} to {target_table}")
                return rows_affected
                
        finally:
            if should_close:
                conn.close()
    
    def upsert_signals(
        self,
        staging_table: str,
        target_table: str,
        signal_columns: List[str],
        conn=None
    ) -> int:
        """
        UPSERT from staging to target signals table.
        
        Args:
            staging_table: Source staging table
            target_table: Target table (e.g., 'kfl.signals_lead')
            signal_columns: List of signal column names
            conn: Optional existing connection
            
        Returns:
            Number of rows affected
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs
                cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
                
                # Build column list
                base_columns = ['time', 'time_close', 'asset_id', 'interval_min', 'source_script']
                all_columns = base_columns + signal_columns
                columns_list = ', '.join(all_columns)
                
                # Update only signal columns
                update_set = ', '.join([
                    f'{col} = EXCLUDED.{col}' 
                    for col in ['time_close', 'source_script'] + signal_columns
                ])
                
                upsert_query = f"""
                    INSERT INTO {target_table} ({columns_list})
                    SELECT {columns_list} FROM {staging_table}
                    ON CONFLICT (time, asset_id, interval_min)
                    DO UPDATE SET {update_set}
                """
                
                cur.execute(upsert_query)
                rows_affected = cur.rowcount
                conn.commit()
                
                logger.info(f"Upserted {rows_affected} rows from {staging_table} to {target_table}")
                return rows_affected
                
        finally:
            if should_close:
                conn.close()
    
    def cleanup_all_staging(self, conn=None):
        """
        Drop all session-specific backfill staging tables.
        
        Args:
            conn: Optional existing connection
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # REASON: Ruim alleen de tabellen op die door DEZE sessie zijn aangemaakt
            tables = self._staging_tables if self._staging_tables else [
                f'staging.indicators_backfill_{self._session_id}',
                f'staging.signals_lead_backfill_{self._session_id}',
                f'staging.signals_coin_backfill_{self._session_id}',
                f'staging.signals_conf_backfill_{self._session_id}',
            ]
            
            with conn.cursor() as cur:
                for table in tables:
                    cur.execute(f"DROP TABLE IF EXISTS {table}")
                conn.commit()
            
            self._staging_tables.clear()
            logger.info(f"Cleaned up staging tables for session {self._session_id}")
            
        finally:
            if should_close:
                conn.close()
