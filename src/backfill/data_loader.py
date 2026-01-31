"""
Data loader for klines with lookback buffer.

Loads klines from kfl.klines_raw with extra rows before start_date
to ensure indicator stability (warmup period).
"""

import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql

from .config import (
    LOOKBACK_BUFFER,
    INTERVAL_MINUTES,
    DEFAULT_DB_CONNECTION,
)

logger = logging.getLogger(__name__)

# Suppress pandas UserWarning about psycopg2 connections
# psycopg2 works perfectly fine with pandas, this is just a pandas warning
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy', category=UserWarning)


class DataLoader:
    """
    Loads klines data with lookback buffer for indicator warmup.
    
    For each asset+interval combination, loads:
    - Lookback buffer: LOOKBACK_BUFFER rows BEFORE start_date
    - Target data: All rows from start_date to end_date
    
    Returns numpy arrays for efficient processing.
    """
    
    def __init__(self, connection_string: str = DEFAULT_DB_CONNECTION):
        """
        Initialize the data loader.

        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string

    def get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(self.connection_string)
    
    def get_available_assets(self, conn=None) -> List[int]:
        """
        Get list of all asset_ids in klines_raw.
        
        Returns:
            List of asset IDs
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT asset_id FROM kfl.klines_raw ORDER BY asset_id")
                return [row[0] for row in cur.fetchall()]
        finally:
            if should_close:
                conn.close()
    
    def get_available_intervals(self, conn=None) -> List[str]:
        """
        Get list of all intervals in klines_raw.
        
        Returns:
            List of interval codes
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT interval_min FROM kfl.klines_raw ORDER BY interval_min")
                return [row[0] for row in cur.fetchall()]
        finally:
            if should_close:
                conn.close()
    
    def calculate_lookback_start(
        self,
        start_date: datetime,
        interval_min: str,
        lookback_rows: int = LOOKBACK_BUFFER
    ) -> datetime:
        """
        Calculate the start time for lookback data.
        
        Args:
            start_date: Target start date
            interval_min: Interval code (e.g., 'D', '240', '1')
            lookback_rows: Number of extra rows to load
            
        Returns:
            Datetime for lookback start
        """
        minutes = INTERVAL_MINUTES.get(interval_min, 1)
        lookback_minutes = minutes * lookback_rows
        return start_date - timedelta(minutes=lookback_minutes)
    
    def load_klines(
        self,
        asset_id: int,
        interval_min: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_lookback: bool = True,
        conn=None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Load klines for a single asset+interval combination.
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            start_date: Start date for target data (None = all)
            end_date: End date for target data (None = now)
            include_lookback: Whether to include lookback buffer before start_date
            conn: Optional existing connection
            
        Returns:
            Tuple of (DataFrame with klines, index where target data starts)
            The DataFrame contains OHLCV columns: time, open, high, low, close, volume
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # Determine query bounds
            if start_date and include_lookback:
                query_start = self.calculate_lookback_start(start_date, interval_min)
            else:
                query_start = start_date
            
            query_end = end_date or datetime.now(timezone.utc)
            
            # Build query
            query = """
                SELECT time, open, high, low, close, volume
                FROM kfl.klines_raw
                WHERE asset_id = %s AND interval_min = %s::kfl.interval_type
            """
            params = [asset_id, interval_min]
            
            if query_start:
                query += " AND time >= %s"
                params.append(query_start)
            
            if query_end:
                query += " AND time <= %s"
                params.append(query_end)
            
            query += " ORDER BY time ASC"

            # Execute query (warning suppressed at module level)
            df = pd.read_sql(query, conn, params=params)
            
            if df.empty:
                logger.debug(f"No klines for asset {asset_id}, interval {interval_min}")
                return df, 0
            
            # Find index where target data starts
            if start_date and include_lookback:
                # Ensure start_date is timezone-aware for comparison
                if start_date.tzinfo is None:
                    start_date_tz = start_date.replace(tzinfo=timezone.utc)
                else:
                    start_date_tz = start_date
                
                # Convert time column to tz-aware if needed
                if df['time'].dt.tz is None:
                    df['time'] = df['time'].dt.tz_localize('UTC')
                
                # Find first row >= start_date
                target_start_idx = df[df['time'] >= start_date_tz].index.min()
                if pd.isna(target_start_idx):
                    target_start_idx = len(df)  # All data is lookback
                else:
                    target_start_idx = df.index.get_loc(target_start_idx)
            else:
                target_start_idx = 0
            
            logger.debug(
                f"Loaded {len(df)} klines for asset {asset_id}/{interval_min}, "
                f"target starts at index {target_start_idx}"
            )
            
            return df, target_start_idx
            
        finally:
            if should_close:
                conn.close()
    
    def load_klines_batch(
        self,
        asset_ids: List[int],
        interval_min: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_lookback: bool = True,
        conn=None
    ) -> Dict[int, Tuple[pd.DataFrame, int]]:
        """
        Load klines for multiple assets at once (same interval).
        
        Args:
            asset_ids: List of asset IDs
            interval_min: Interval code
            start_date: Start date for target data
            end_date: End date for target data
            include_lookback: Whether to include lookback buffer
            conn: Optional existing connection
            
        Returns:
            Dict mapping asset_id -> (DataFrame, target_start_idx)
        """
        results = {}
        
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            for asset_id in asset_ids:
                df, idx = self.load_klines(
                    asset_id=asset_id,
                    interval_min=interval_min,
                    start_date=start_date,
                    end_date=end_date,
                    include_lookback=include_lookback,
                    conn=conn
                )
                if not df.empty:
                    results[asset_id] = (df, idx)
        finally:
            if should_close:
                conn.close()
        
        return results
    
    def to_numpy_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convert DataFrame to numpy arrays for GPU/CPU processing.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            Dict with keys: time, open, high, low, close, volume
            All values are numpy float64 arrays (time as int64 nanoseconds)
        """
        return {
            'time': df['time'].values.astype('datetime64[ns]').astype(np.int64),
            'open': df['open'].values.astype(np.float64),
            'high': df['high'].values.astype(np.float64),
            'low': df['low'].values.astype(np.float64),
            'close': df['close'].values.astype(np.float64),
            'volume': df['volume'].values.astype(np.float64),
        }
    
    def get_time_range(
        self,
        asset_id: int,
        interval_min: str,
        conn=None
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get min and max time for an asset+interval.
        
        Returns:
            Tuple of (min_time, max_time) or (None, None) if no data
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MIN(time), MAX(time)
                    FROM kfl.klines_raw
                    WHERE asset_id = %s AND interval_min = %s::kfl.interval_type
                """, (asset_id, interval_min))
                row = cur.fetchone()
                return (row[0], row[1]) if row else (None, None)
        finally:
            if should_close:
                conn.close()
    
    def load_klines_from_gap(
        self,
        asset_id: int,
        interval_min: str,
        gap_time: datetime,
        end_date: Optional[datetime] = None,
        lookback_rows: int = LOOKBACK_BUFFER,
        conn=None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Load klines starting from (gap_time - lookback) to end_date.
        
        Optimized for gap-forward mode: loads only the data needed to fill gaps,
        plus a lookback buffer for indicator warmup.
        
        Args:
            asset_id: Asset ID
            interval_min: Interval code
            gap_time: First gap timestamp (from find_first_gap)
            end_date: End date for target data (None = now)
            lookback_rows: Number of rows before gap_time to load for warmup
            conn: Optional existing connection
            
        Returns:
            Tuple of (DataFrame with klines, index where new data starts)
            The index points to the first row at or after gap_time.
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            # Calculate lookback start time
            lookback_start = self.calculate_lookback_start(gap_time, interval_min, lookback_rows)
            query_end = end_date or datetime.now(timezone.utc)
            
            # Build query
            query = """
                SELECT time, open, high, low, close, volume
                FROM kfl.klines_raw
                WHERE asset_id = %s 
                    AND interval_min = %s::kfl.interval_type
                    AND time >= %s
                    AND time <= %s
                ORDER BY time ASC
            """
            params = [asset_id, interval_min, lookback_start, query_end]
            
            df = pd.read_sql(query, conn, params=params)
            
            if df.empty:
                logger.debug(f"No klines for asset {asset_id}, interval {interval_min} from gap")
                return df, 0
            
            # Ensure gap_time is timezone-aware for comparison
            if gap_time.tzinfo is None:
                gap_time_tz = gap_time.replace(tzinfo=timezone.utc)
            else:
                gap_time_tz = gap_time
            
            # Convert time column to tz-aware if needed
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('UTC')
            
            # Find index where gap_time starts (first row >= gap_time)
            target_start_idx = df[df['time'] >= gap_time_tz].index.min()
            if pd.isna(target_start_idx):
                target_start_idx = len(df)  # All data is lookback
            else:
                target_start_idx = df.index.get_loc(target_start_idx)
            
            logger.debug(
                f"Loaded {len(df)} klines from gap for asset {asset_id}/{interval_min}, "
                f"new data starts at index {target_start_idx}"
            )
            
            return df, target_start_idx
            
        finally:
            if should_close:
                conn.close()
    
    def count_klines(
        self,
        asset_id: Optional[int] = None,
        interval_min: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        conn=None
    ) -> int:
        """
        Count klines matching the criteria.
        
        Returns:
            Number of matching rows
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        try:
            query = "SELECT COUNT(*) FROM kfl.klines_raw WHERE 1=1"
            params = []
            
            if asset_id is not None:
                query += " AND asset_id = %s"
                params.append(asset_id)
            
            if interval_min is not None:
                query += " AND interval_min = %s::kfl.interval_type"
                params.append(interval_min)
            
            if start_date:
                query += " AND time >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND time <= %s"
                params.append(end_date)
            
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()[0]
        finally:
            if should_close:
                conn.close()
