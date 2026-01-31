"""
MTF (Multi-TimeFrame) Backfill Writer using Python-based joins.

Builds MTF signal records from historical signal tables without lookahead bias.
Uses Python/pandas for efficient O(N) hash joins instead of SQL JOINs that cause
Cartesian product explosions on TimescaleDB hypertables.

ARCHITECTURE:
1. Fetch 1m base data for a time chunk
2. Fetch D/4H/1H data for the same period (with lookback buffer)
3. Calculate time_bucket values in Python
4. Merge using pandas hash join (O(N) vs SQL's O(N×chunks))
5. Bulk INSERT with COPY for maximum throughput

LOOKAHEAD BIAS PREVENTION:
- time_bucket() returns the START of the containing bucket
- Example: base.time_close=01:30, time_bucket('1 hour', 01:30) = 01:00
- We join on tf_60.time_close = 01:00
- The 1H candle with time_close=01:00 is the 00:00-01:00 candle
- That candle closed at 01:00, which is BEFORE 01:30 ✓ No lookahead bias

UTC ALIGNMENT ASSUMPTION:
- Assumes candles are aligned to UTC bucket boundaries
- Daily: 00:00 UTC, 4H: 00:00/04:00/08:00/..., 1H: every hour

INCOMPLETE DATA HANDLING:
- Rows where higher TF candles don't exist yet are SKIPPED
- This happens at the very start of a dataset (before first daily candle closes)
"""

import io
import logging
import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

from ..config import (
    DEFAULT_DB_CONNECTION, SIGNAL_COLS, 
    MTF_MAX_WORKERS, MTF_CHUNK_DAYS, MTF_FETCH_SIZE
)

# REASON: Suppress pandas warning about psycopg2 - werkt prima, alleen niet officieel getest
warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')

logger = logging.getLogger(__name__)


class MTFBackfillWriter:
    """
    MTF backfill via PostgreSQL time_bucket() set-based joins.
    
    Performance: O(N+M) vs O(N×M) voor LATERAL JOIN.
    Lookahead-vrij: gebruikt time_bucket() om alleen afgesloten candles te joinen.
    Een signaal is pas BEKEND op time_close, niet op time.
    """
    
    # Timeframe mapping: classification -> (source_table, target_table, columns)
    # REASON: ALLE kolommen (bool, discrete, concordance) krijgen timeframe suffix
    TABLE_CONFIG = {
        'lead': {
            'source': 'kfl.signals_lead',
            'target': 'kfl.mtf_signals_lead',
            'bool_columns': list(SIGNAL_COLS.LEAD),  # Krijgen _d, _240, _60, _1 suffix
            'discrete_columns': list(SIGNAL_COLS.LEAD_DISCRETE),  # Krijgen _d, _240, _60, _1 suffix
            'concordance': list(SIGNAL_COLS.CONCORDANCE),  # Krijgen _d, _240, _60, _1 suffix
        },
        'coin': {
            'source': 'kfl.signals_coin',
            'target': 'kfl.mtf_signals_coin',
            'bool_columns': list(SIGNAL_COLS.COIN),
            'discrete_columns': list(SIGNAL_COLS.COIN_DISCRETE),
            'concordance': list(SIGNAL_COLS.CONCORDANCE),
        },
        'conf': {
            'source': 'kfl.signals_conf',
            'target': 'kfl.mtf_signals_conf',
            'bool_columns': list(SIGNAL_COLS.CONF),
            'discrete_columns': list(SIGNAL_COLS.CONF_DISCRETE),
            'concordance': list(SIGNAL_COLS.CONCORDANCE),
        },
    }
    
    # MTF timeframes: suffix -> interval_min code
    TIMEFRAMES = {
        'd': 'D',      # Daily (structural)
        '240': '240',  # 4H (tactical)
        '60': '60',    # 1H (entry)
        '1': '1',      # 1m (utf/micro)
    }
    
    @staticmethod
    def calculate_optimal_workers(max_workers: Optional[int] = None) -> int:
        """
        Bereken optimaal aantal workers voor MTF backfill.
        
        MTF backfill is I/O bound (database queries), niet CPU intensief.
        Aanbeveling: 2-8 workers afhankelijk van systeem resources.
        
        Args:
            max_workers: Optioneel maximum. Als None, wordt optimaal berekend.
            
        Returns:
            Optimaal aantal workers (2-8, of user-specified 1-16)
        """
        if max_workers is not None:
            # User specified: cap tussen 1-16
            return max(1, min(16, max_workers))
        
        # Auto-calculate: I/O bound, gebruik ~50% van CPU cores
        cpu_count = multiprocessing.cpu_count()
        optimal = max(2, min(8, cpu_count // 2))
        return optimal
    
    def __init__(self, connection_string: str = DEFAULT_DB_CONNECTION):
        """
        Initialize MTF backfill writer.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
    
    def get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(self.connection_string)
    
    def _build_bucket_join_query(
        self,
        classification: str,
        asset_id: int,
        start_date: datetime,
        end_date: datetime,
        intervals: Optional[List[str]] = None
    ) -> Tuple[str, Tuple]:
        """
        Build set-based bucket join query for MTF backfill.
        
        REASON: Expliciete tijdsgrenzen toegevoegd aan JOINs voor chunk exclusion.
        Zonder tijdsgrenzen moet PostgreSQL ALLE hypertable chunks scannen per JOIN,
        wat resulteert in O(chunks^3) complexiteit = miljarden row comparisons.
        
        Met tijdsgrenzen kan TimescaleDB chunk exclusion toepassen:
        - Daily candles: start_date - 1 dag tot end_date (voor lookback)
        - 4H candles: start_date - 4 uur tot end_date
        - 1H candles: start_date - 1 uur tot end_date
        """
        config = self.TABLE_CONFIG[classification]
        source_table = config['source']
        target_table = config['target']
        bool_cols = config['bool_columns']
        discrete_cols = config['discrete_columns']
        concordance_cols = config['concordance']
        
        # Default naar alle timeframes als niets is opgegeven
        if not intervals:
            intervals = ['1', '60', '240', 'D']
        
        # Alleen intervallen die we ondersteunen in de MTF tabel
        active_tfs = []
        if 'D' in intervals: active_tfs.append('d')
        if '240' in intervals: active_tfs.append('240')
        if '60' in intervals: active_tfs.append('60')
        if '1' in intervals: active_tfs.append('1')
        
        # ALL signal columns (bool + discrete + concordance) - all get TF suffix
        all_signal_cols = bool_cols + discrete_cols + concordance_cols
        
        # Build column lists for INSERT
        time_cols = []
        for tf in active_tfs:
            time_cols.extend([f'time_{tf}', f'time_close_{tf}'])
        
        # ALL signal columns per timeframe (signal_name_d, signal_name_240, etc.)
        mtf_signal_cols = []
        for col in all_signal_cols:
            for tf in active_tfs:
                mtf_signal_cols.append(f'{col}_{tf}')
        
        # Build SELECT expressions
        select_time_cols = []
        select_signal_cols = []
        
        joins = []
        where_filters = []
        
        # REASON: Bereken lookback voor chunk exclusion
        # Daily: we hebben data nodig voor time_bucket('1 day', x) waar x in [start_date, end_date)
        # De time_bucket output voor start_date kan 1 dag eerder zijn
        # Query parameters: start_date, end_date, plus lookback bounds per timeframe
        
        # Base is altijd de laagste TF geselecteerd, of 1m als '1' in intervals
        # Voor nu gaan we ervan uit dat base altijd '1' is (zoals in de tabel structuur)
        if '1' in active_tfs:
            select_time_cols.extend(['base.time AS time_1', 'base.time_close AS time_close_1'])
            for col in all_signal_cols:
                select_signal_cols.append(f'base.{col} AS {col}_1')

        # REASON: Expliciete tijdsgrenzen in JOINs voor chunk exclusion
        # De time_close van hogere TFs moet binnen het bereik liggen van de base chunk
        # met een kleine buffer voor time_bucket alignment
        if 'd' in active_tfs:
            select_time_cols.extend(['tf_d.time AS time_d', 'tf_d.time_close AS time_close_d'])
            for col in all_signal_cols:
                select_signal_cols.append(f'tf_d.{col} AS {col}_d')
            # REASON: time_close constraint beperkt scan tot relevante chunks
            joins.append(f"""
            LEFT JOIN {source_table} tf_d
                ON tf_d.asset_id = base.asset_id
                AND tf_d.interval_min = 'D'
                AND tf_d.time_close = time_bucket('1 day', base.time_close)
                AND tf_d.time_close >= %s - INTERVAL '2 days'
                AND tf_d.time_close < %s + INTERVAL '1 day'
            """)
            where_filters.append("tf_d.time_close IS NOT NULL")

        if '240' in active_tfs:
            select_time_cols.extend(['tf_240.time AS time_240', 'tf_240.time_close AS time_close_240'])
            for col in all_signal_cols:
                select_signal_cols.append(f'tf_240.{col} AS {col}_240')
            joins.append(f"""
            LEFT JOIN {source_table} tf_240
                ON tf_240.asset_id = base.asset_id
                AND tf_240.interval_min = '240'
                AND tf_240.time_close = time_bucket('4 hours', base.time_close)
                AND tf_240.time_close >= %s - INTERVAL '8 hours'
                AND tf_240.time_close < %s + INTERVAL '4 hours'
            """)
            where_filters.append("tf_240.time_close IS NOT NULL")

        if '60' in active_tfs:
            select_time_cols.extend(['tf_60.time AS time_60', 'tf_60.time_close AS time_close_60'])
            for col in all_signal_cols:
                select_signal_cols.append(f'tf_60.{col} AS {col}_60')
            joins.append(f"""
            LEFT JOIN {source_table} tf_60
                ON tf_60.asset_id = base.asset_id
                AND tf_60.interval_min = '60'
                AND tf_60.time_close = time_bucket('1 hour', base.time_close)
                AND tf_60.time_close >= %s - INTERVAL '2 hours'
                AND tf_60.time_close < %s + INTERVAL '1 hour'
            """)
            where_filters.append("tf_60.time_close IS NOT NULL")
        
        # REASON: CTE met ROW_NUMBER() voorkomt "cannot affect row a second time" error
        # door expliciet 1 rij per (asset_id, time) te selecteren voordat INSERT plaatsvindt.
        query = f"""
        WITH ranked_data AS (
            SELECT
                base.asset_id,
                {', '.join(select_time_cols)},
                {', '.join(select_signal_cols)},
                ROW_NUMBER() OVER (PARTITION BY base.asset_id, base.time ORDER BY base.time) as rn
            FROM {source_table} base
            {' '.join(joins)}
            WHERE base.interval_min = '1'
              AND base.asset_id = %s
              AND base.time >= %s
              AND base.time < %s
              { 'AND ' + ' AND '.join(where_filters) if where_filters else '' }
        )
        INSERT INTO {target_table} (
            asset_id,
            {', '.join(time_cols)},
            {', '.join(mtf_signal_cols)}
        )
        SELECT
            asset_id,
            {', '.join(time_cols)},
            {', '.join(mtf_signal_cols)}
        FROM ranked_data
        WHERE rn = 1
        
        ON CONFLICT (asset_id, time_1) DO UPDATE SET
            {', '.join([f'{col} = EXCLUDED.{col}' for col in time_cols])},
            {', '.join([f'{col} = EXCLUDED.{col}' for col in mtf_signal_cols])}
        """
        
        # REASON: Build parameters - elke JOIN krijgt start/end voor chunk exclusion
        # Volgorde: (D start, D end, 240 start, 240 end, 60 start, 60 end, asset_id, start, end)
        params = []
        if 'd' in active_tfs:
            params.extend([start_date, end_date])
        if '240' in active_tfs:
            params.extend([start_date, end_date])
        if '60' in active_tfs:
            params.extend([start_date, end_date])
        params.extend([asset_id, start_date, end_date])
        
        return query, tuple(params)
    
    def _fetch_signals_data(
        self,
        classification: str,
        asset_id: int,
        interval_min: str,
        start_date: datetime,
        end_date: datetime,
        conn
    ) -> pd.DataFrame:
        """
        Fetch signals data for one asset/interval/timerange.
        
        REASON: Gebruikt server-side cursor met fetch batches om RAM te beperken
        op zowel client als server. MTF_FETCH_SIZE bepaalt batch grootte.
        
        Returns DataFrame with time_close as index for efficient merging.
        """
        config = self.TABLE_CONFIG[classification]
        source_table = config['source']
        all_signal_cols = (
            list(config['bool_columns']) + 
            list(config['discrete_columns']) + 
            list(config['concordance'])
        )
        
        cols_sql = ', '.join(all_signal_cols)
        query = f"""
            SELECT time, time_close, {cols_sql}
            FROM {source_table}
            WHERE asset_id = %s
              AND interval_min = %s
              AND time_close >= %s
              AND time_close < %s
            ORDER BY time_close
        """
        
        # REASON: Server-side cursor voorkomt dat PostgreSQL hele resultset
        # in shared_buffers laadt. Fetch in batches van MTF_FETCH_SIZE.
        if MTF_FETCH_SIZE > 0:
            chunks = []
            cursor_name = f'mtf_fetch_{classification}_{asset_id}_{interval_min}'
            
            with conn.cursor(name=cursor_name) as cur:
                cur.itersize = MTF_FETCH_SIZE
                cur.execute(query, (asset_id, interval_min, start_date, end_date))
                
                # REASON: Bij named cursors is description pas beschikbaar NA eerste fetch
                colnames = None
                while True:
                    rows = cur.fetchmany(MTF_FETCH_SIZE)
                    if not rows:
                        break
                    # Haal column names op bij eerste batch
                    if colnames is None:
                        colnames = [desc[0] for desc in cur.description]
                    chunks.append(pd.DataFrame(rows, columns=colnames))
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.DataFrame(columns=['time', 'time_close'] + all_signal_cols)
        else:
            # Fallback: load all at once (legacy behavior)
            df = pd.read_sql(query, conn, params=(asset_id, interval_min, start_date, end_date))
        
        return df
    
    def _time_bucket(self, timestamps: pd.Series, interval: str) -> pd.Series:
        """
        Calculate time_bucket equivalent in Python.
        Returns the START of the bucket containing each timestamp.
        
        REASON: Python implementation is O(N) vs SQL time_bucket in JOIN which causes
        Cartesian product when PostgreSQL can't optimize.
        """
        if interval == 'D':
            # Daily: truncate to start of day
            return timestamps.dt.floor('D')
        elif interval == '240':
            # 4 hours: truncate to 4-hour boundary (0, 4, 8, 12, 16, 20)
            # REASON: '4h' ipv '4H' - pandas deprecation warning fix
            return timestamps.dt.floor('4h')
        elif interval == '60':
            # 1 hour: truncate to hour
            # REASON: 'h' ipv 'H' - pandas deprecation warning fix
            return timestamps.dt.floor('h')
        else:
            return timestamps
    
    def _upsert_mtf_data(
        self,
        classification: str,
        df: pd.DataFrame,
        asset_id: int,
        conn
    ) -> int:
        """
        Upsert MTF data using efficient COPY + temp table approach.
        
        Returns number of rows upserted.
        """
        if df.empty:
            return 0
        
        config = self.TABLE_CONFIG[classification]
        target_table = config['target']
        
        # REASON: Fix dtype issues na pandas merge - concordance kolommen worden
        # soms object dtype door NaN handling, maar moeten float zijn voor PostgreSQL
        for col in df.columns:
            if 'concordance_score' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif 'concordance_sum' in col or 'concordance_count' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Get column names from dataframe
        columns = list(df.columns)
        
        # Create temp table
        temp_table = f"_mtf_temp_{classification}_{asset_id}"
        
        with conn.cursor() as cur:
            # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs
            cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
            
            # Drop temp table if exists
            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
            
            # Create temp table with same structure
            cols_def = ', '.join([f'"{c}" {self._get_pg_type(df[c].dtype)}' for c in columns])
            cur.execute(f"CREATE TEMP TABLE {temp_table} ({cols_def})")
            
            # Use COPY for fast bulk insert into temp table
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, header=False, na_rep='\\N')
            buffer.seek(0)
            
            cur.copy_expert(
                f"COPY {temp_table} ({', '.join([f'\"{c}\"' for c in columns])}) FROM STDIN WITH CSV NULL '\\N'",
                buffer
            )
            
            # Upsert from temp to target
            # Build ON CONFLICT update list (exclude primary key columns)
            pk_cols = ['asset_id', 'time_1']
            update_cols = [c for c in columns if c not in pk_cols]
            
            if update_cols:
                update_set = ', '.join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])
                upsert_sql = f"""
                    INSERT INTO {target_table} ({', '.join([f'"{c}"' for c in columns])})
                    SELECT {', '.join([f'"{c}"' for c in columns])} FROM {temp_table}
                    ON CONFLICT (asset_id, time_1) DO UPDATE SET {update_set}
                """
            else:
                upsert_sql = f"""
                    INSERT INTO {target_table} ({', '.join([f'"{c}"' for c in columns])})
                    SELECT {', '.join([f'"{c}"' for c in columns])} FROM {temp_table}
                    ON CONFLICT (asset_id, time_1) DO NOTHING
                """
            
            cur.execute(upsert_sql)
            rows = cur.rowcount
            
            # Cleanup
            cur.execute(f"DROP TABLE {temp_table}")
            conn.commit()
            
            return rows
    
    def _get_pg_type(self, dtype) -> str:
        """Map pandas dtype to PostgreSQL type."""
        dtype_str = str(dtype).lower()  # REASON: Case-insensitive voor Int64/int64
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'DOUBLE PRECISION'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMPTZ'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        else:
            return 'TEXT'
    
    def backfill_classification(
        self,
        classification: str,
        asset_id: int,
        start_date: datetime,
        end_date: datetime,
        intervals: Optional[List[str]] = None,
        chunk_days: int = 7,
        conn=None,
        show_progress: bool = True,
        parent_pbar: Optional[tqdm] = None
    ) -> int:
        """
        Backfill MTF for one classification using Python-based joins.
        
        REASON: SQL JOINs with time_bucket() cause Cartesian product on TimescaleDB.
        Python hash joins are O(N) and much faster.
        
        Process:
        1. Fetch 1m base data
        2. Fetch D/4H/1H data for same period  
        3. Calculate time_bucket in Python
        4. Merge using pandas (hash join)
        5. Bulk upsert result
        """
        should_close = conn is None
        if conn is None:
            conn = self.get_connection()
        
        if not intervals:
            intervals = ['1', '60', '240', 'D']
        
        config = self.TABLE_CONFIG[classification]
        all_signal_cols = (
            list(config['bool_columns']) + 
            list(config['discrete_columns']) + 
            list(config['concordance'])
        )
        
        total_rows = 0
        
        try:
            # Add lookback buffer for higher timeframes
            fetch_start = start_date - timedelta(days=2)
            
            logger.debug(f"Fetching {classification} data for asset {asset_id}, {start_date} to {end_date}")
            
            # Fetch all timeframe data
            dfs = {}
            for interval in intervals:
                df = self._fetch_signals_data(
                    classification, asset_id, interval, 
                    fetch_start, end_date, conn
                )
                if not df.empty:
                    # Ensure time_close is datetime
                    df['time_close'] = pd.to_datetime(df['time_close'], utc=True)
                    dfs[interval] = df
                    logger.debug(f"  {interval}: {len(df)} rows")
            
            # Need at least 1m data
            if '1' not in dfs or dfs['1'].empty:
                logger.warning(f"No 1m data for {classification} asset {asset_id}")
                return 0
            
            # Start with 1m base data (only within actual date range)
            base = dfs['1'].copy()
            base = base[(base['time_close'] >= start_date) & (base['time_close'] < end_date)]
            
            if base.empty:
                return 0
            
            # Rename columns with _1 suffix
            base = base.rename(columns={
                'time': 'time_1',
                'time_close': 'time_close_1',
                **{col: f'{col}_1' for col in all_signal_cols}
            })
            
            # Add asset_id
            base['asset_id'] = asset_id
            
            # Calculate bucket keys for joining
            base['bucket_d'] = self._time_bucket(base['time_close_1'], 'D')
            base['bucket_240'] = self._time_bucket(base['time_close_1'], '240')
            base['bucket_60'] = self._time_bucket(base['time_close_1'], '60')
            
            # Merge with Daily data
            if 'D' in dfs and not dfs['D'].empty:
                df_d = dfs['D'].copy()
                df_d = df_d.rename(columns={
                    'time': 'time_d',
                    'time_close': 'time_close_d',
                    **{col: f'{col}_d' for col in all_signal_cols}
                })
                # Join on bucket_d = time_close_d
                base = base.merge(
                    df_d,
                    left_on='bucket_d',
                    right_on='time_close_d',
                    how='inner'  # Skip rows without daily data
                )
            else:
                logger.warning(f"No Daily data for {classification} asset {asset_id}")
                return 0
            
            # Merge with 4H data
            if '240' in dfs and not dfs['240'].empty:
                df_240 = dfs['240'].copy()
                df_240 = df_240.rename(columns={
                    'time': 'time_240',
                    'time_close': 'time_close_240',
                    **{col: f'{col}_240' for col in all_signal_cols}
                })
                base = base.merge(
                    df_240,
                    left_on='bucket_240',
                    right_on='time_close_240',
                    how='inner'
                )
            else:
                logger.warning(f"No 4H data for {classification} asset {asset_id}")
                return 0
            
            # Merge with 1H data
            if '60' in dfs and not dfs['60'].empty:
                df_60 = dfs['60'].copy()
                df_60 = df_60.rename(columns={
                    'time': 'time_60',
                    'time_close': 'time_close_60',
                    **{col: f'{col}_60' for col in all_signal_cols}
                })
                base = base.merge(
                    df_60,
                    left_on='bucket_60',
                    right_on='time_close_60',
                    how='inner'
                )
            else:
                logger.warning(f"No 1H data for {classification} asset {asset_id}")
                return 0
            
            # Drop bucket columns (not needed in target table)
            base = base.drop(columns=['bucket_d', 'bucket_240', 'bucket_60'], errors='ignore')
            
            # Reorder columns to match target table structure
            # Order: asset_id, time_d, time_close_d, time_240, time_close_240, time_60, time_close_60, time_1, time_close_1, signals...
            time_cols = [
                'asset_id',
                'time_d', 'time_close_d',
                'time_240', 'time_close_240', 
                'time_60', 'time_close_60',
                'time_1', 'time_close_1'
            ]
            
            signal_cols = []
            for col in all_signal_cols:
                for tf in ['d', '240', '60', '1']:
                    signal_cols.append(f'{col}_{tf}')
            
            # Only include columns that exist
            final_cols = [c for c in time_cols + signal_cols if c in base.columns]
            base = base[final_cols]
            
            # REASON: Source tracking voor backfill
            base['source_script'] = 'GPU_backfill'
            
            logger.debug(f"Merged data: {len(base)} rows, {len(base.columns)} columns")
            
            # Upsert to target table
            if not base.empty:
                total_rows = self._upsert_mtf_data(classification, base, asset_id, conn)
                logger.debug(f"Upserted {total_rows} rows for {classification}")
            
            return total_rows
            
        except Exception as e:
            logger.error(f"Error in backfill_classification {classification} asset {asset_id}: {e}", exc_info=True)
            raise
        finally:
            if should_close:
                conn.close()
    
    def backfill_asset(
        self,
        asset_id: int,
        start_date: Optional[datetime],
        end_date: datetime,
        intervals: Optional[List[str]] = None,
        chunk_days: int = 10,  # REASON: Default 10 dagen voor veiligheid
        show_progress: bool = True,
        gap_fill: bool = False
    ) -> Dict[str, int]:
        """
        Backfill all MTF tables for one asset.
        REASON: Toegevoegd 'gap_fill' om per asset de eerste gap te zoeken.
        REASON: Chunking toegevoegd om RAM-vriendelijk te blijven per asset.
        """
        results = {'lead': 0, 'coin': 0, 'conf': 0}
        classifications = list(self.TABLE_CONFIG.keys())
        
        # REASON: In gap_fill modus zoeken we hier de eerste gap voor deze specifieke asset
        actual_start = start_date
        if gap_fill:
            asset_gap = self.find_first_gap([asset_id], end_date=end_date)
            if asset_gap:
                # 1 dag extra lookback voor alignment/warmup
                actual_start = (asset_gap - timedelta(days=1)).replace(tzinfo=timezone.utc)
                logger.info(f"Asset {asset_id}: MTF gap gevonden op {asset_gap.strftime('%Y-%m-%d %H:%M')}, start op {actual_start.strftime('%Y-%m-%d %H:%M')}")
            else:
                # REASON: Geen gap gevonden, forceer 48 uur correctie
                actual_start = (datetime.now(timezone.utc) - timedelta(hours=48))
                logger.info(f"Asset {asset_id}: Geen MTF gaps, geforceerde correctie vanaf {actual_start.strftime('%Y-%m-%d %H:%M')}")
            
            if start_date and actual_start < start_date:
                actual_start = start_date

        if not actual_start:
            logger.warning(f"Asset {asset_id}: Geen startdatum bepaald, skip.")
            return results

        # REASON: Verdeel de periode in chunks om RAM-spikes te voorkomen
        current_start = actual_start
        effective_chunk_days = chunk_days if chunk_days > 0 else 30 # Fallback naar 30 dagen als 0
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=effective_chunk_days), end_date)
            
            conn = self.get_connection()
            try:
                if show_progress:
                    print(f"\n📊 Asset {asset_id} Chunk: {current_start.strftime('%Y-%m-%d')} tot {current_end.strftime('%Y-%m-%d')}")
                
                for classification in classifications:
                    rows = self.backfill_classification(
                        classification, asset_id, current_start, current_end,
                        intervals=intervals, chunk_days=0, conn=conn, show_progress=False
                    )
                    results[classification] += rows
                    
                    if show_progress and rows > 0:
                        print(f"  ✅ {classification}: {rows:,} rows")
                
                current_start = current_end
                
            finally:
                conn.close()
        
        return results
    
    def get_earliest_kline_date(self, asset_id: int, interval: str = '1') -> Optional[datetime]:
        """
        Haal de vroegste kline datum op voor een asset en interval.
        
        Args:
            asset_id: Asset ID
            interval: Interval (default '1' voor 1m)
            
        Returns:
            Vroegste kline datetime of None als geen data
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                # Zoek in signals tabel (not indicators) want daar staan de resulterende signalen
                cur.execute("""
                    SELECT MIN(time) 
                    FROM kfl.signals_lead 
                    WHERE asset_id = %s AND interval_min = %s
                """, (asset_id, interval))
                result = cur.fetchone()
                return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"Fout bij ophalen vroegste kline voor asset {asset_id}: {e}", exc_info=True)
            return None
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    def get_earliest_dates_batch(self, asset_ids: List[int], interval: str = '1') -> Dict[int, datetime]:
        """
        Haal vroegste kline datum op voor meerdere assets in één query.
        
        REASON: Efficiënter dan N individuele queries - reduceert roundtrips.
        
        Args:
            asset_ids: Lijst van asset IDs
            interval: Interval (default '1' voor 1m)
            
        Returns:
            Dict van {asset_id: earliest_datetime}, alleen assets met data
        """
        if not asset_ids:
            return {}
        
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                # REASON: Batch query met ANY() is efficiënter dan N losse queries
                cur.execute("""
                    SELECT asset_id, MIN(time) as earliest
                    FROM kfl.signals_lead 
                    WHERE asset_id = ANY(%s) AND interval_min = %s
                    GROUP BY asset_id
                """, (asset_ids, interval))
                
                result = {}
                for row in cur.fetchall():
                    if row[1]:  # Skip NULL values
                        result[row[0]] = row[1]
                
                logger.info(f"Earliest dates opgehaald: {len(result)}/{len(asset_ids)} assets hebben 1m data")
                return result
        except Exception as e:
            logger.error(f"Fout bij batch ophalen vroegste klines: {e}", exc_info=True)
            return {}
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    def calculate_chunks(
        self,
        asset_ids: List[int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        chunk_months: int = 2,
        chunk_days: int = 0
    ) -> List[Tuple[datetime, datetime, List[int]]]:
        """
        Bereken chunks op basis van vroegste kline per asset.
        REASON: Voortgang in chunks, niet per asset. Chunk size in dagen of maanden.
        
        Args:
            asset_ids: Lijst van asset IDs
            start_date: Expliciete start datum (optioneel, anders vroegste kline)
            end_date: Eind datum (default: nu)
            chunk_months: Chunk grootte in maanden (default 2, ignored als chunk_days > 0)
            chunk_days: Chunk grootte in dagen (0 = gebruik chunk_months)
            
        Returns:
            List van (chunk_start, chunk_end, assets_in_chunk) tuples
        """
        from dateutil.relativedelta import relativedelta
        from datetime import timedelta
        
        logger.debug(f"calculate_chunks: start_date={start_date}, end_date={end_date}, chunk_months={chunk_months}, chunk_days={chunk_days}")
        
        if end_date is None:
            end_date = datetime.now(timezone.utc)
            logger.debug(f"end_date set to now: {end_date}")
        
        # Bepaal overall start datum
        if start_date is None:
            logger.info("Start datum niet opgegeven, zoeken naar vroegste kline...")
            # Zoek de vroegste kline over alle assets
            earliest = None
            for asset_id in asset_ids:
                logger.debug(f"  Ophalen vroegste kline voor asset {asset_id}...")
                asset_earliest = self.get_earliest_kline_date(asset_id, '1')
                if asset_earliest:
                    logger.debug(f"  Asset {asset_id}: vroegste = {asset_earliest}")
                    if earliest is None or asset_earliest < earliest:
                        earliest = asset_earliest
                else:
                    logger.debug(f"  Asset {asset_id}: geen data gevonden")
            
            if earliest is None:
                logger.warning("Geen klines gevonden voor geselecteerde assets")
                return []
            
            start_date = earliest
            logger.info(f"Vroegste kline gevonden: {start_date}")
        
        # Maak chunks van N dagen of N maanden
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            if chunk_days > 0:
                # REASON: Dagen-modus voor fijnere RAM-controle
                chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            else:
                chunk_end = min(current_start + relativedelta(months=chunk_months), end_date)
            
            # Alle assets in deze chunk
            chunks.append((current_start, chunk_end, asset_ids))
            
            current_start = chunk_end
        
        return chunks
    
    def _process_single_asset(
        self,
        asset_id: int,
        start_date: Optional[datetime],
        end_date: datetime,
        intervals: List[str],
        gap_fill: bool = False,
        chunk_days: int = 10
    ) -> Tuple[int, Dict[str, int]]:
        """
        Process een enkele asset - helper voor parallelle verwerking.
        REASON: ThreadPoolExecutor heeft een callable nodig die alle args meekrijgt.
        """
        logger.debug(f"Worker starting asset {asset_id}...")
        try:
            result = self.backfill_asset(
                asset_id, start_date, end_date,
                intervals=intervals,
                chunk_days=chunk_days,
                show_progress=False,
                gap_fill=gap_fill
            )
            logger.debug(f"Worker finished asset {asset_id}: {sum(result.values())} rows")
            return (asset_id, result)
        except Exception as e:
            logger.error(f"Asset {asset_id} failed: {e}", exc_info=True)
            return (asset_id, {'error': str(e)})
    
    def backfill_parallel(
        self,
        asset_ids: List[int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        intervals: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        chunk_months: int = 0,
        chunk_days: int = 10,  # REASON: Default 10 dagen is RAM-vriendelijker dan 1 maand
        show_progress: bool = True,
        gap_fill: bool = False
    ) -> Dict[int, Dict[str, int]]:
        """
        Parallelle backfill voor multiple assets.
        
        REASON: Per-asset verwerking heeft GEEN race conditions - elke asset
        heeft unieke rijen (PK = asset_id + time_1). Parallellisatie is veilig.
        """
        import sys
        
        logger.info(f"backfill_parallel() gestart: {len(asset_ids)} assets, intervals={intervals}, gap_fill={gap_fill}")
        
        results = {aid: {'lead': 0, 'coin': 0, 'conf': 0} for aid in asset_ids}
        
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        if not intervals:
            intervals = ['1', '60', '240', 'D']
        
        # Bepaal start_date (vroegste kline als niet opgegeven)
        # REASON: In gap_fill modus doen we dit per asset in backfill_asset
        if start_date is None and not gap_fill:
            logger.info("Zoeken naar vroegste kline voor gehele set...")
            # ... (bestaande logica voor overall start_date)
            earliest = None
            for asset_id in asset_ids:
                asset_earliest = self.get_earliest_kline_date(asset_id, '1')
                if asset_earliest:
                    if earliest is None or asset_earliest < earliest:
                        earliest = asset_earliest
            
            if earliest is None:
                logger.warning("Geen klines gevonden")
                return results
            
            start_date = earliest
            logger.info(f"Vroegste kline: {start_date}")
        
        # Bepaal aantal workers
        if max_workers is None:
            max_workers = MTF_MAX_WORKERS
        actual_workers = min(max_workers, len(asset_ids))
        
        period_msg = f"Periode: {start_date.date() if start_date else 'Earliest'} tot {end_date.date()}"
        assets_msg = f"Assets: {len(asset_ids)}"
        workers_msg = f"Workers: {actual_workers}"
        
        # FAST MODE: alleen als BEIDE chunk_months=0 EN chunk_days=0
        if chunk_months == 0 and chunk_days == 0:
            logger.warning(f"⚠️  FAST MODE: hele periode per asset - gebruik chunk_months>0 voor grote datasets!")
            if show_progress:
                print(f"\n⚠️  FAST MODE: {actual_workers} workers parallel (RAM-intensief!)")
                print(f"📅 {period_msg}")
                print(f"📊 {assets_msg}")
            
            total_rows_all = 0
            tqdm_enabled = show_progress and (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty())
            
            pbar = None
            if show_progress:
                try:
                    pbar = tqdm(total=len(asset_ids), desc="MTF Assets", unit="asset", ncols=100, disable=not tqdm_enabled)
                except Exception: pbar = None
            
            # PARALLEL EXECUTION
            if actual_workers > 1:
                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    futures = {
                        executor.submit(
                            self._process_single_asset,
                            aid, start_date, end_date, intervals, gap_fill, chunk_days
                        ): aid for aid in asset_ids
                    }
                    
                    completed_count = 0
                    for future in as_completed(futures):
                        aid = futures[future]
                        completed_count += 1
                        try:
                            asset_id, asset_result = future.result()
                            if 'error' not in asset_result:
                                asset_rows = sum(asset_result.values())
                                for classification, rows in asset_result.items():
                                    results[asset_id][classification] += rows
                                total_rows_all += asset_rows
                                logger.info(f"  Asset {asset_id}: {asset_rows:,} rows ({completed_count}/{len(asset_ids)})")
                            else:
                                results[asset_id] = asset_result
                        except Exception as e:
                            logger.error(f"Asset {aid} future failed: {e}")
                            results[aid] = {'error': str(e)}
                        
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix_str(f"{total_rows_all:,} rows")
            else:
                # Sequential
                for aid in asset_ids:
                    _, asset_result = self._process_single_asset(aid, start_date, end_date, intervals, gap_fill)
                    if 'error' not in asset_result:
                        for classification, rows in asset_result.items():
                            results[aid][classification] += rows
                        total_rows_all += sum(asset_result.values())
                    if pbar: pbar.update(1)
            
            if pbar: pbar.close()
        
        else:
            # CHUNK MODE (default)
            # REASON: In gap_fill modus doen we GEEN chunking over assets heen, 
            # maar verwerken we elke asset individueel (met zijn eigen gap).
            if gap_fill:
                logger.info("📦 GAP-FILL MODE: Verwerking per asset (individuele gaps)")
                total_rows_all = 0
                tqdm_enabled = show_progress and (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty())
                
                pbar = None
                if show_progress:
                    try:
                        pbar = tqdm(total=len(asset_ids), desc="MTF Assets", unit="asset", ncols=100, disable=not tqdm_enabled)
                    except Exception: pbar = None

                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    futures = {
                        executor.submit(
                            self._process_single_asset,
                            aid, start_date, end_date, intervals, True, chunk_days
                        ): aid for aid in asset_ids
                    }
                    for future in as_completed(futures):
                        aid, asset_result = future.result()
                        if 'error' not in asset_result:
                            for classification, rows in asset_result.items():
                                results[aid][classification] += rows
                            total_rows_all += sum(asset_result.values())
                        
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix_str(f"{total_rows_all:,} rows")
                
                if pbar: pbar.close()
                return results

            # ... (Rest van de bestaande CHUNK MODE logica voor full runs) ...
            logger.info("📦 CHUNK MODE: RAM-vriendelijk, berekenen van chunks...")
            
            # REASON: Pre-fetch earliest dates per asset om lege chunks te skippen
            # Dit voorkomt duizenden "No 1m data" warnings voor assets die pas later data hebben
            logger.info("📊 Ophalen eerste data datum per asset...")
            earliest_dates = self.get_earliest_dates_batch(asset_ids, '1')
            
            # Filter assets die helemaal geen data hebben
            assets_with_data = [aid for aid in asset_ids if aid in earliest_dates]
            assets_without_data = [aid for aid in asset_ids if aid not in earliest_dates]
            
            if assets_without_data:
                logger.warning(f"⚠️  {len(assets_without_data)} assets hebben geen 1m data: {assets_without_data[:10]}{'...' if len(assets_without_data) > 10 else ''}")
            
            if not assets_with_data:
                logger.warning("Geen assets met 1m data gevonden")
                return results
            
            try:
                chunks = self.calculate_chunks(assets_with_data, start_date, end_date, chunk_months, chunk_days)
            except Exception as e:
                logger.error(f"Fout bij calculate_chunks: {e}", exc_info=True)
                raise
            
            if not chunks:
                logger.warning("Geen chunks om te verwerken")
                return results
            
            # REASON: Dynamische chunk beschrijving voor dagen of maanden
            if chunk_days > 0:
                chunks_msg = f"Chunks: {len(chunks)} × {chunk_days} dag(en)"
                chunk_desc = f"{chunk_days} dag(en)"
            else:
                chunks_msg = f"Chunks: {len(chunks)} × {chunk_months} maand(en)"
                chunk_desc = f"{chunk_months} maand(en)"
            
            logger.info(period_msg)
            logger.info(chunks_msg)
            logger.info(f"Assets met data: {len(assets_with_data)}")
            logger.info(workers_msg)
            
            if show_progress:
                print(f"\n📦 CHUNK MODE: {chunk_desc} per chunk, {actual_workers} workers")
                print(f"📅 {period_msg}")
                print(f"📦 {chunks_msg}")
                print(f"📊 Assets met data: {len(assets_with_data)}/{len(asset_ids)}")
            
            total_rows_all = 0
            skipped_asset_chunks = 0
            tqdm_enabled = show_progress and (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty())
            
            pbar = None
            if show_progress:
                try:
                    pbar = tqdm(
                        total=len(chunks),
                        desc="MTF Chunks",
                        unit="chunk",
                        ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                        disable=not tqdm_enabled
                    )
                except Exception:
                    pbar = None
            
            # REASON: Parallelle verwerking per chunk met worker limiet
            # Elke chunk bevat alle assets, we verwerken chunks sequentieel
            # maar assets binnen een chunk parallel
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_start, chunk_end, chunk_assets = chunk_data
                
                # REASON: Filter assets die data hebben VOOR chunk_end
                # Skip assets waarvan earliest_date >= chunk_end (nog geen data in deze chunk)
                eligible_assets = [
                    aid for aid in chunk_assets 
                    if aid in earliest_dates and earliest_dates[aid] < chunk_end
                ]
                
                skipped_count = len(chunk_assets) - len(eligible_assets)
                if skipped_count > 0:
                    skipped_asset_chunks += skipped_count
                    logger.debug(f"Chunk {chunk_idx + 1}: {skipped_count} assets geskipt (geen data vóór {chunk_end.date()})")
                
                if not eligible_assets:
                    logger.debug(f"Chunk {chunk_idx + 1}/{len(chunks)}: SKIP - geen assets met data")
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix_str(f"{total_rows_all:,} rows, {skipped_asset_chunks} skipped")
                    continue
                
                logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)}: {chunk_start.date()} - {chunk_end.date()} ({len(eligible_assets)} assets)")
                
                if actual_workers > 1 and len(eligible_assets) > 1:
                    # Parallel binnen chunk
                    def process_chunk_asset(aid):
                        try:
                            return (aid, self.backfill_asset(
                                aid, chunk_start, chunk_end, 
                                intervals=intervals, 
                                chunk_days=0,  # Geen sub-chunking binnen maand-chunk
                                show_progress=False
                            ))
                        except Exception as e:
                            logger.error(f"Asset {aid} failed: {e}")
                            return (aid, {'error': str(e)})
                    
                    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                        futures = {executor.submit(process_chunk_asset, aid): aid for aid in eligible_assets}
                        
                        for future in as_completed(futures):
                            try:
                                aid, asset_result = future.result()
                                if 'error' not in asset_result:
                                    for classification, rows in asset_result.items():
                                        results[aid][classification] += rows
                                        total_rows_all += rows
                                else:
                                    results[aid] = asset_result
                            except Exception as e:
                                aid = futures[future]
                                logger.error(f"Asset {aid} future failed: {e}")
                                results[aid] = {'error': str(e)}
                else:
                    # Sequential
                    for aid in eligible_assets:
                        try:
                            asset_result = self.backfill_asset(
                                aid, chunk_start, chunk_end, 
                                intervals=intervals, 
                                chunk_days=0,
                                show_progress=False
                            )
                            
                            if 'error' not in asset_result:
                                for classification, rows in asset_result.items():
                                    results[aid][classification] += rows
                                    total_rows_all += rows
                                    
                        except Exception as e:
                            logger.error(f"Chunk {chunk_start.date()}-{chunk_end.date()} asset {aid} failed: {e}", exc_info=True)
                            results[aid] = {'error': str(e)}
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"{total_rows_all:,} rows")
            
            if pbar:
                pbar.close()
            
            # REASON: Log summary van geskipte asset-chunks
            if skipped_asset_chunks > 0:
                logger.info(f"📊 Totaal {skipped_asset_chunks} asset-chunk combinaties geskipt (geen data in die periode)")
        
        if show_progress:
            success_assets = len([
                r for r in results.values() 
                if isinstance(r, dict) and 'error' not in r and all(isinstance(v, (int, float)) and v >= 0 for v in r.values())
            ])
            print(f"\n✅ Voltooid: {success_assets}/{len(asset_ids)} assets, {total_rows_all:,} totaal rows")
        
        return results
    
    def verify_no_lookahead_bias(
        self,
        classification: str,
        asset_id: int,
        sample_size: int = 1000
    ) -> Dict[str, any]:
        """
        Verify that MTF backfill has no lookahead bias.
        
        Checks that no MTF row contains signals from candles that
        weren't closed yet at the time of the 1m candle.
        
        Args:
            classification: 'lead', 'coin', or 'conf'
            asset_id: Asset ID to verify
            sample_size: Number of rows to check
            
        Returns:
            Dict with verification results
        """
        config = self.TABLE_CONFIG[classification]
        target_table = config['target']
        source_table = config['source']
        
        # Check: no 60m source candle should have time_close > mtf.time_close_1
        verify_query = f"""
        SELECT COUNT(*) as violations
        FROM {target_table} mtf
        WHERE mtf.asset_id = %s
          AND (
            mtf.time_close_d > mtf.time_close_1
            OR mtf.time_close_240 > mtf.time_close_1
            OR mtf.time_close_60 > mtf.time_close_1
          )
        LIMIT %s
        """
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(verify_query, (asset_id, sample_size))
                violations = cur.fetchone()[0]
                
                # Get sample count
                cur.execute(f"SELECT COUNT(*) FROM {target_table} WHERE asset_id = %s", (asset_id,))
                total_rows = cur.fetchone()[0]
            
            return {
                'classification': classification,
                'asset_id': asset_id,
                'total_rows': total_rows,
                'violations': violations,
                'passed': violations == 0,
                'message': 'No lookahead bias detected' if violations == 0 else f'{violations} rows with lookahead bias!'
            }
            
        finally:
            conn.close()

    def find_first_gap(self, asset_ids: List[int], end_date: Optional[datetime] = None) -> Optional[datetime]:
        """
        Zoek de vroegste datum waar MTF data ontbreekt vergeleken met signals_lead.
        
        REASON: Maakt 'gap-fill' modus mogelijk zonder handmatige datum invoer.
        REASON: Gaps in de eerste 32 dagen negeren (warmup periode).
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Vind vroegste kline per asset om threshold te bepalen
                # We gebruiken signals_lead als proxy voor klines_raw hier
                query = """
                    WITH asset_thresholds AS (
                        SELECT asset_id, MIN(time) + INTERVAL '32 days' as threshold
                        FROM kfl.signals_lead
                        WHERE asset_id = ANY(%s) AND interval_min = '1'
                        GROUP BY asset_id
                    )
                    SELECT MIN(s.time)
                    FROM kfl.signals_lead s
                    JOIN asset_thresholds t ON s.asset_id = t.asset_id
                    LEFT JOIN kfl.mtf_signals_lead m ON s.asset_id = m.asset_id AND s.time = m.time_1
                    WHERE s.asset_id = ANY(%s)
                      AND s.interval_min = '1'
                      AND s.time >= t.threshold
                      AND m.time_1 IS NULL
                """
                params = [asset_ids, asset_ids]
                if end_date:
                    query += " AND s.time <= %s"
                    params.append(end_date)
                
                cur.execute(query, tuple(params))
                return cur.fetchone()[0]
        finally:
            conn.close()


def create_mtf_indexes(connection_string: str = DEFAULT_DB_CONNECTION) -> bool:
    """
    Create indexes required for efficient MTF bucket join queries.
    
    Index: (asset_id, interval_min, time_close DESC)
    Optimizes: time_bucket() based LEFT JOIN lookups for MTF backfill.
    
    Note: Uses regular CREATE INDEX (not CONCURRENTLY) because
    hypertables do not support concurrent index creation.
    
    Args:
        connection_string: PostgreSQL connection string
        
    Returns:
        True if successful
    """
    conn = psycopg2.connect(connection_string)
    
    try:
        with conn.cursor() as cur:
            for table in ['signals_lead', 'signals_coin', 'signals_conf']:
                index_name = f'idx_{table}_mtf_lookup'
                logger.info(f"Creating index {index_name}...")
                
                # REASON: Hypertables don't support CREATE INDEX CONCURRENTLY
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON kfl.{table} (asset_id, interval_min, time_close DESC)
                """)
            
            conn.commit()
            logger.info("MTF indexes created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to create MTF indexes: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()
