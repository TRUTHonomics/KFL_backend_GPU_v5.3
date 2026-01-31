"""
Gap detector for identifying missing timestamps in target tables.

Used in 'gaps_only' mode to skip already-computed rows.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psycopg2

from .config import DEFAULT_DB_CONNECTION

logger = logging.getLogger(__name__)


class GapDetector:
    """
    Detects gaps (missing timestamps) in indicator and signal tables.
    
    Used to determine which rows need to be computed in 'gaps_only' mode.
    """
    
    def __init__(self, connection_string: str = DEFAULT_DB_CONNECTION):
        """
        Initialize the gap detector.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
    
    def get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(self.connection_string)
    
    def get_existing_timestamps(
        self,
        table: str,
        asset_id: int,
        interval_min: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        conn=None
    ) -> Set[datetime]:
        """
        Get set of existing timestamps in a target table.
        
        Args:
            table: Table name (e.g., 'kfl.indicators', 'kfl.signals_lead')
            asset_id: Asset ID
            interval_min: Interval code
            start_date: Optional start date filter
            end_date: Optional end date filter
            conn: Optional existing connection
            
        Returns:
            Set of existing timestamps
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            query = f"""
                SELECT time FROM {table}
                WHERE asset_id = %s AND interval_min = %s::kfl.interval_type
            """
            params = [asset_id, interval_min]
            
            if start_date:
                query += " AND time >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND time <= %s"
                params.append(end_date)
            
            with conn.cursor() as cur:
                cur.execute(query, params)
                return {row[0] for row in cur.fetchall()}
                
        finally:
            if should_close:
                conn.close()
    
    def create_write_mask(
        self,
        kline_times: np.ndarray,
        existing_timestamps: Set[datetime],
        target_start_idx: int
    ) -> np.ndarray:
        """
        Create a boolean mask indicating which rows to write.
        
        Args:
            kline_times: Array of timestamps from klines (as datetime64 or datetime)
            existing_timestamps: Set of timestamps already in target table
            target_start_idx: Index where target data starts (after lookback buffer)
            
        Returns:
            Boolean numpy array where True = needs to be written
        """
        n = len(kline_times)
        mask = np.zeros(n, dtype=bool)
        
        # Only check rows after the lookback buffer
        for i in range(target_start_idx, n):
            ts = kline_times[i]
            # Convert numpy datetime64 to python datetime if needed
            if isinstance(ts, np.datetime64):
                ts = pd.Timestamp(ts).to_pydatetime()
            
            if ts not in existing_timestamps:
                mask[i] = True
        
        return mask
    
    def detect_indicator_gaps(
        self,
        asset_id: int,
        interval_min: str,
        kline_times: np.ndarray,
        target_start_idx: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        conn=None
    ) -> np.ndarray:
        """
        Detect gaps in kfl.indicators table.
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            kline_times: Array of timestamps from klines
            target_start_idx: Index where target data starts
            start_date: Optional start date filter
            end_date: Optional end date filter
            conn: Optional existing connection
            
        Returns:
            Boolean mask where True = row is missing in indicators
        """
        existing = self.get_existing_timestamps(
            table='kfl.indicators',
            asset_id=asset_id,
            interval_min=interval_min,
            start_date=start_date,
            end_date=end_date,
            conn=conn
        )
        
        return self.create_write_mask(kline_times, existing, target_start_idx)
    
    def detect_signal_gaps(
        self,
        asset_id: int,
        interval_min: str,
        kline_times: np.ndarray,
        target_start_idx: int,
        table: str = 'kfl.signals_lead',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        conn=None
    ) -> np.ndarray:
        """
        Detect gaps in a signals table.
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            kline_times: Array of timestamps from klines
            target_start_idx: Index where target data starts
            table: Signal table name
            start_date: Optional start date filter
            end_date: Optional end date filter
            conn: Optional existing connection
            
        Returns:
            Boolean mask where True = row is missing in signals
        """
        existing = self.get_existing_timestamps(
            table=table,
            asset_id=asset_id,
            interval_min=interval_min,
            start_date=start_date,
            end_date=end_date,
            conn=conn
        )
        
        return self.create_write_mask(kline_times, existing, target_start_idx)
    
    def detect_all_gaps(
        self,
        asset_id: int,
        interval_min: str,
        kline_times: np.ndarray,
        target_start_idx: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        conn=None
    ) -> Dict[str, np.ndarray]:
        """
        Detect gaps in all target tables.
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            kline_times: Array of timestamps from klines
            target_start_idx: Index where target data starts
            start_date: Optional start date filter
            end_date: Optional end date filter
            conn: Optional existing connection
            
        Returns:
            Dict mapping table name -> boolean write mask
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            tables = [
                'kfl.indicators',
                'kfl.signals_lead',
                'kfl.signals_coin',
                'kfl.signals_conf',
            ]
            
            results = {}
            for table in tables:
                existing = self.get_existing_timestamps(
                    table=table,
                    asset_id=asset_id,
                    interval_min=interval_min,
                    start_date=start_date,
                    end_date=end_date,
                    conn=conn
                )
                results[table] = self.create_write_mask(
                    kline_times, existing, target_start_idx
                )
            
            return results
            
        finally:
            if should_close:
                conn.close()
    
    def count_gaps(
        self,
        asset_id: int,
        interval_min: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        conn=None
    ) -> Dict[str, int]:
        """
        Count number of gaps per table.
        
        Compares klines_raw with target tables to find missing rows.
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            start_date: Optional start date filter
            end_date: Optional end date filter
            conn: Optional existing connection
            
        Returns:
            Dict mapping table name -> number of missing rows
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # Get klines timestamps
            query = """
                SELECT time FROM kfl.klines_raw
                WHERE asset_id = %s AND interval_min = %s::kfl.interval_type
            """
            params = [asset_id, interval_min]
            
            if start_date:
                query += " AND time >= %s"
                params.append(start_date)
            if end_date:
                query += " AND time <= %s"
                params.append(end_date)
            
            with conn.cursor() as cur:
                cur.execute(query, params)
                kline_times = {row[0] for row in cur.fetchall()}
            
            if not kline_times:
                return {}
            
            # Compare with each target table
            tables = [
                'kfl.indicators',
                'kfl.signals_lead',
                'kfl.signals_coin',
                'kfl.signals_conf',
            ]
            
            results = {}
            for table in tables:
                existing = self.get_existing_timestamps(
                    table=table,
                    asset_id=asset_id,
                    interval_min=interval_min,
                    start_date=start_date,
                    end_date=end_date,
                    conn=conn
                )
                missing = kline_times - existing
                results[table] = len(missing)
            
            return results
            
        finally:
            if should_close:
                conn.close()
    
    def find_first_gap(
        self,
        asset_id: int,
        interval_min: str,
        end_date: Optional[datetime] = None,
        conn=None
    ) -> Optional[datetime]:
        """
        Find the first timestamp in klines_raw that is missing from indicators.
        
        Uses indicators as proxy for all derived tables (signals are written together).
        This is much more efficient than loading all timestamps - it stops at the first gap.
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            end_date: Optional end date filter
            conn: Optional existing connection
            
        Returns:
            First gap timestamp, or None if data is complete.
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # REASON: Gaps in de eerste 32 dagen negeren (warmup periode)
            with conn.cursor() as cur:
                # 1. Vind de allereerste kline
                cur.execute("""
                    SELECT MIN(time) FROM kfl.klines_raw 
                    WHERE asset_id = %s AND interval_min = %s::kfl.interval_type
                """, (asset_id, interval_min))
                earliest_kline = cur.fetchone()[0]
                
                if not earliest_kline:
                    return None
                
                # 2. Threshold is 32 dagen na de start
                from datetime import timedelta
                gap_threshold = earliest_kline + timedelta(days=32)
                
                # 3. Zoek eerste gap vanaf threshold
                # REASON: LEFT JOIN met IS NULL check vindt eerste ontbrekende rij
                # ORDER BY + LIMIT 1 stopt zodra eerste gap gevonden is
                query = """
                    SELECT k.time
                    FROM kfl.klines_raw k
                    LEFT JOIN kfl.indicators i 
                        ON k.asset_id = i.asset_id 
                        AND k.interval_min = i.interval_min 
                        AND k.time = i.time
                    WHERE k.asset_id = %s 
                        AND k.interval_min = %s::kfl.interval_type
                        AND k.time >= %s
                        AND i.time IS NULL
                """
                params = [asset_id, interval_min, gap_threshold]
                
                if end_date:
                    query += " AND k.time <= %s"
                    params.append(end_date)
                
                query += " ORDER BY k.time ASC LIMIT 1"
                
                cur.execute(query, params)
                row = cur.fetchone()
                
                if row is None:
                    return None  # Data is complete (na de 32 dagen warmup)
                
                return row[0]
                
        finally:
            if should_close:
                conn.close()
    
    def get_gap_summary(
        self,
        asset_ids: Optional[List[int]] = None,
        intervals: Optional[List[str]] = None,
        conn=None
    ) -> pd.DataFrame:
        """
        Get summary of gaps across multiple assets/intervals.
        
        Returns:
            DataFrame with columns: asset_id, interval_min, table, gaps
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # Get all asset/interval combinations if not specified
            if asset_ids is None:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT asset_id FROM kfl.klines_raw ORDER BY asset_id")
                    asset_ids = [row[0] for row in cur.fetchall()]
            
            if intervals is None:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT interval_min FROM kfl.klines_raw")
                    intervals = [row[0] for row in cur.fetchall()]
            
            rows = []
            for asset_id in asset_ids:
                for interval_min in intervals:
                    gaps = self.count_gaps(asset_id, interval_min, conn=conn)
                    for table, count in gaps.items():
                        if count > 0:
                            rows.append({
                                'asset_id': asset_id,
                                'interval_min': interval_min,
                                'table': table,
                                'gaps': count
                            })
            
            return pd.DataFrame(rows)
            
        finally:
            if should_close:
                conn.close()
