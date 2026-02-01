#!/usr/bin/env python3
"""
DEPRECATED: Dit script is niet meer in gebruik.
ATR/Outcome backfill is verplaatst naar QBN_v2 container.

Gebruik in plaats hiervan: QBN_v2/scripts/outcome_backfill.py
Outcomes worden nu centraal opgeslagen in qbn.signal_outcomes (hypertable).

========================================================================
ORIGINELE DOCUMENTATIE (DEPRECATED):
========================================================================

Client-Side ATR Backfill Script v2 voor KFL Backend GPU v5.3

ARCHITECTUUR:
Deze versie haalt data OP naar de Windows machine (96GB RAM, 9950X3D CPU) en
doet de joins daar met pandas, in plaats van zware JOINs op de database server.

Dit voorkomt OOM crashes op VM120 (48GB) bij JOIN van grote hypertables.

FLOW:
1. Fetch chunk van mtf_signals_* waar atr_at_signal IS NULL
2. Fetch overeenkomstige atr_14 waarden uit kfl.indicators  
3. Pandas hash merge (O(N) - zeer efficiënt)
4. Bulk UPDATE via COPY + temp table

REASON: De oude atr_backfill.py deed UPDATE...FROM...JOIN queries op de database,
wat 4 workers x 100M row joins = OOM op 48GB server.
Deze versie gebruikt max 96GB RAM op Windows client, DB server blijft vrij.

USAGE:
    python -m src.backfill.scripts.atr_backfill_v2 --workers 8 --table all
    python -m src.backfill.scripts.atr_backfill_v2 --table lead --chunk-size 500000
"""

import sys
import os
import io
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# KFL Backend imports
from ..config import DEFAULT_DB_CONNECTION, ATR_MAX_WORKERS

# KFL logregels: _log/ met setup_kfl_logging
from pathlib import Path
_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))
from utils.kfl_logging import setup_kfl_logging

logger = setup_kfl_logging("atr_backfill_v2", log_level=logging.INFO)

# MTF tabellen mapping
MTF_TABLES = {
    'lead': 'kfl.mtf_signals_lead',
    'coin': 'kfl.mtf_signals_coin', 
    'conf': 'kfl.mtf_signals_conf'
}

# REASON: Verlaagd naar 100.000 voor stabielere RAM-druk op DB server
DEFAULT_CHUNK_SIZE = 100_000  # 100K rows per chunk
DEFAULT_FETCH_SIZE = 100_000  # Server-side cursor batch size


class ATRBackfillClientSide:
    """
    Client-side ATR backfill - haalt data naar Windows, doet joins lokaal.
    
    REASON: Vergelijkbaar met MTF backfill pattern dat succesvol draaide
    met slechts 1GB RAM gebruik terwijl het miljoenen rows verwerkte.
    """
    
    def __init__(self, connection_string: str = DEFAULT_DB_CONNECTION):
        self.connection_string = connection_string
        
    def get_connection(self):
        """Get a new database connection."""
        return psycopg2.connect(self.connection_string)

    def find_first_gap(self, table_name: str, end_date: Optional[datetime] = None) -> Optional[datetime]:
        """
        Zoek de vroegste datum waar atr_at_signal NULL is.
        
        REASON: Maakt automatische gap-fill mogelijk.
        REASON: Gaps in de eerste 32 dagen negeren (warmup periode).
        """
        table = MTF_TABLES[table_name]
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Vind vroegste data per asset + 32 dagen threshold
                # 2. Vind eerste NULL vanaf die threshold
                query = f"""
                    WITH asset_thresholds AS (
                        SELECT asset_id, MIN(time_1) + INTERVAL '32 days' as threshold
                        FROM {table}
                        GROUP BY asset_id
                    )
                    SELECT MIN(mtf.time_1) 
                    FROM {table} mtf
                    JOIN asset_thresholds t ON mtf.asset_id = t.asset_id
                    WHERE mtf.atr_at_signal IS NULL
                      AND mtf.time_1 >= t.threshold
                """
                params = []
                if end_date:
                    query += " AND mtf.time_1 <= %s"
                    params.append(end_date)
                
                cur.execute(query, tuple(params))
                return cur.fetchone()[0]
        finally:
            conn.close()
    
    def get_null_count(self, table: str) -> Tuple[int, int, int]:
        """
        Get counts voor een MTF tabel.
        
        Returns:
            Tuple van (total, null_count, filled_count)
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE atr_at_signal IS NULL) as null_count,
                        COUNT(*) FILTER (WHERE atr_at_signal IS NOT NULL) as filled_count
                    FROM {table}
                """)
                row = cur.fetchone()
                return row[0], row[1], row[2]
        finally:
            conn.close()
    
    def get_time_ranges(self, table: str, num_chunks: int, start_date: Optional[datetime] = None) -> List[Tuple[datetime, datetime]]:
        """
        Bepaal time ranges voor chunked processing.
        
        REASON: Time-based chunking werkt beter met TimescaleDB hypertables
        omdat het chunk exclusion mogelijk maakt.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                if start_date:
                    # In gap-fill modus beginnen we vanaf start_date tot nu
                    min_time = start_date
                    cur.execute(f"SELECT MAX(time_1) FROM {table}")
                    max_time = cur.fetchone()[0]
                else:
                    cur.execute(f"SELECT MIN(time_1), MAX(time_1) FROM {table} WHERE atr_at_signal IS NULL")
                    row = cur.fetchone()
                    min_time, max_time = row[0], row[1]
                
                if min_time is None or max_time is None:
                    return []
                
                # Verdeel in chunks
                time_diff = max_time - min_time
                chunk_interval = time_diff / num_chunks if num_chunks > 0 else time_diff
                
                ranges = []
                current = min_time
                for i in range(num_chunks):
                    if i == num_chunks - 1:
                        ranges.append((current, max_time))
                    else:
                        end = current + chunk_interval
                        ranges.append((current, end))
                        current = end
                
                return ranges
        finally:
            conn.close()
    
    def _fetch_mtf_nulls(
        self,
        table: str,
        time_start: datetime,
        time_end: datetime,
        conn,
        gap_fill: bool = False
    ) -> pd.DataFrame:
        """
        Fetch MTF records voor processing.
        
        REASON: In gap_fill=True modus halen we per asset data op vanaf de EERSTE gap van die asset.
        Dit is veel efficiënter dan vanaf de globale eerste gap van de hele tabel.
        """
        if gap_fill:
            # REASON: Subquery vindt de oudste gap per asset_id
            # Gaps in de eerste 32 dagen worden genegeerd.
            # Als er geen gap is (NULL), forceren we 48 uur geleden als startpunt.
            query = f"""
                WITH asset_stats AS (
                    SELECT 
                        asset_id, 
                        MIN(time_1) + INTERVAL '32 days' as threshold
                    FROM {table}
                    GROUP BY asset_id
                ),
                asset_gaps AS (
                    SELECT 
                        m.asset_id, 
                        COALESCE(MIN(m.time_1), NOW() - INTERVAL '48 hours') as first_gap
                    FROM {table} m
                    JOIN asset_stats s ON m.asset_id = s.asset_id
                    WHERE m.atr_at_signal IS NULL
                      AND m.time_1 >= s.threshold
                    GROUP BY m.asset_id
                )
                SELECT mtf.asset_id, mtf.time_1
                FROM {table} mtf
                JOIN asset_gaps g ON mtf.asset_id = g.asset_id
                WHERE mtf.time_1 >= g.first_gap
                  AND mtf.time_1 >= %s
                  AND mtf.time_1 <= %s
                ORDER BY mtf.asset_id, mtf.time_1
            """
        else:
            # Haal alleen NULL records op
            query = f"""
                SELECT asset_id, time_1
                FROM {table}
                WHERE atr_at_signal IS NULL
                  AND time_1 >= %s
                  AND time_1 <= %s
                ORDER BY asset_id, time_1
            """
        
        # REASON: Server-side cursor voor efficiënte streaming
        chunks = []
        # REASON: Gebruik abs() voor hash want hyphens zijn ongeldig in SQL identifiers
        cursor_name = f'atr_mtf_fetch_{abs(hash((table, str(time_start))))}'
        
        with conn.cursor(name=cursor_name) as cur:
            cur.itersize = DEFAULT_FETCH_SIZE
            cur.execute(query, (time_start, time_end))
            
            colnames = None
            while True:
                rows = cur.fetchmany(DEFAULT_FETCH_SIZE)
                if not rows:
                    break
                if colnames is None:
                    colnames = [desc[0] for desc in cur.description]
                chunks.append(pd.DataFrame(rows, columns=colnames))
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame(columns=['asset_id', 'time_1'])
    
    def _fetch_indicators(
        self,
        asset_ids: List[int],
        time_start: datetime,
        time_end: datetime,
        conn
    ) -> pd.DataFrame:
        """
        Fetch ATR waarden uit kfl.indicators voor gegeven assets en tijdsbereik.
        
        REASON: Haalt alleen relevante kolommen op. Filtert op asset_ids
        om data transfer te minimaliseren.
        """
        if not asset_ids:
            return pd.DataFrame(columns=['asset_id', 'time', 'atr_14'])
        
        # REASON: ANY() met array is efficiënter dan grote IN clause
        query = """
            SELECT asset_id, time, atr_14
            FROM kfl.indicators
            WHERE asset_id = ANY(%s)
              AND interval_min = '1'::kfl.interval_type
              AND time >= %s
              AND time <= %s
              AND atr_14 IS NOT NULL
              AND atr_14 > 0
            ORDER BY asset_id, time
        """
        
        chunks = []
        # REASON: Gebruik abs() voor hash want hyphens zijn ongeldig in SQL identifiers
        cursor_name = f'atr_ind_fetch_{abs(hash(str(time_start)))}'
        
        with conn.cursor(name=cursor_name) as cur:
            cur.itersize = DEFAULT_FETCH_SIZE
            cur.execute(query, (asset_ids, time_start, time_end))
            
            colnames = None
            while True:
                rows = cur.fetchmany(DEFAULT_FETCH_SIZE)
                if not rows:
                    break
                if colnames is None:
                    colnames = [desc[0] for desc in cur.description]
                chunks.append(pd.DataFrame(rows, columns=colnames))
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame(columns=['asset_id', 'time', 'atr_14'])
    
    def _bulk_update(
        self,
        table: str,
        df: pd.DataFrame,
        conn,
        gap_fill: bool = False
    ) -> int:
        """
        Bulk update ATR waarden via COPY + temp table pattern.
        
        REASON: Dit is 10-100x sneller dan individuele UPDATEs.
        In gap_fill=True overschrijven we ook bestaande (instabiele) data.
        """
        if df.empty:
            return 0
        
        # Maak temp table
        # REASON: Gebruik abs() voor hash want hyphens zijn ongeldig in SQL identifiers
        temp_table = f"_atr_update_temp_{abs(hash(str(datetime.now())))}"
        
        with conn.cursor() as cur:
            # REASON: TimescaleDB decompressie limiet opheffen voor grote UPSERTs (nodig voor gap-fill)
            cur.execute("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0")
            
            # Drop temp table if exists (safety)
            cur.execute(f"DROP TABLE IF EXISTS {temp_table}")
            
            # Create temp table
            cur.execute(f"""
                CREATE TEMP TABLE {temp_table} (
                    asset_id INTEGER,
                    time_1 TIMESTAMPTZ,
                    atr_at_signal DOUBLE PRECISION
                )
            """)
            
            # COPY data naar temp table
            buffer = io.StringIO()
            df[['asset_id', 'time_1', 'atr_at_signal']].to_csv(
                buffer, index=False, header=False, na_rep='\\N'
            )
            buffer.seek(0)
            
            cur.copy_expert(
                f"COPY {temp_table} (asset_id, time_1, atr_at_signal) FROM STDIN WITH CSV NULL '\\N'",
                buffer
            )
            
            # Update main table from temp table
            # REASON: In gap-fill modus laten we de 'atr_at_signal IS NULL' check weg
            where_atr_null = "AND mtf.atr_at_signal IS NULL" if not gap_fill else ""
            
            cur.execute(f"""
                UPDATE {table} mtf
                SET atr_at_signal = tmp.atr_at_signal,
                    source_script = 'GPU_backfill'
                FROM {temp_table} tmp
                WHERE mtf.asset_id = tmp.asset_id
                  AND mtf.time_1 = tmp.time_1
                  {where_atr_null}
            """)
            updated = cur.rowcount
            
            # Cleanup
            cur.execute(f"DROP TABLE {temp_table}")
            conn.commit()
            
            return updated

    def process_chunk(
        self,
        table: str,
        time_start: datetime,
        time_end: datetime,
        chunk_id: int,
        gap_fill: bool = False
    ) -> Tuple[int, int, float]:
        """
        Verwerk één time chunk: fetch -> merge -> update.
        
        Returns:
            Tuple van (fetched_count, updated_count, duration_seconds)
        """
        start_time = time.time()
        conn = self.get_connection()
        
        try:
            # 1. Fetch MTF records
            logger.debug(f"Chunk {chunk_id}: Fetching MTF records...")
            df_mtf = self._fetch_mtf_nulls(table, time_start, time_end, conn, gap_fill=gap_fill)
            
            if df_mtf.empty:
                logger.debug(f"Chunk {chunk_id}: No records found")
                return 0, 0, time.time() - start_time
            
            fetched = len(df_mtf)
            
            # 2. Get unique asset_ids voor indicator query
            asset_ids = df_mtf['asset_id'].unique().tolist()
            
            # 3. Fetch indicators
            logger.debug(f"Chunk {chunk_id}: Fetching indicators for {len(asset_ids)} assets...")
            df_ind = self._fetch_indicators(asset_ids, time_start, time_end, conn)
            
            if df_ind.empty:
                logger.debug(f"Chunk {chunk_id}: No indicators found")
                return fetched, 0, time.time() - start_time
            
            # 4. Pandas merge (hash join - O(N))
            # REASON: Dit is de kern van de optimalisatie - join gebeurt in 96GB RAM
            logger.debug(f"Chunk {chunk_id}: Merging {len(df_mtf)} MTF with {len(df_ind)} indicators...")
            
            df_merged = df_mtf.merge(
                df_ind,
                left_on=['asset_id', 'time_1'],
                right_on=['asset_id', 'time'],
                how='inner'
            )
            
            if df_merged.empty:
                logger.debug(f"Chunk {chunk_id}: No matches found")
                return fetched, 0, time.time() - start_time
            
            # 5. Prepare update dataframe
            df_update = df_merged[['asset_id', 'time_1', 'atr_14']].copy()
            df_update = df_update.rename(columns={'atr_14': 'atr_at_signal'})
            
            # 6. Bulk update
            logger.debug(f"Chunk {chunk_id}: Updating {len(df_update)} records...")
            updated = self._bulk_update(table, df_update, conn, gap_fill=gap_fill)
            
            duration = time.time() - start_time
            logger.debug(f"Chunk {chunk_id}: Done - {fetched} fetched, {updated} updated in {duration:.1f}s")
            
            return fetched, updated, duration
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} failed: {e}", exc_info=True)
            return 0, 0, time.time() - start_time
        finally:
            conn.close()
    
    def backfill_table(
        self,
        table_name: str,
        num_workers: int = ATR_MAX_WORKERS,
        num_chunks: int = 100,
        show_progress: bool = True,
        gap_fill: bool = False,
        start_date: Optional[datetime] = None
    ) -> Dict:
        """
        Backfill ATR voor één MTF tabel met client-side joins.
        
        Args:
            table_name: Tabel naam (lead/coin/conf)
            num_workers: Aantal parallel workers
            num_chunks: Aantal time-based chunks
            show_progress: Toon progress bar
            gap_fill: Als True, begin vanaf eerste gap en overschrijf data
            start_date: Expliciete start datum voor volledige herberekening (overschrijft gap_fill)
        """
        table = MTF_TABLES[table_name]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ATR Backfill v2 (Client-Side) voor {table}")
        if start_date:
            logger.info(f"Mode: FULL (overschrijft ALLES vanaf {start_date.strftime('%Y-%m-%d %H:%M')})")
            gap_fill = True # Full mode impliceert overschrijven
        else:
            logger.info(f"Mode: {'GAP-FILL (overschrijft instabiele data)' if gap_fill else 'NORMAL (vult alleen NULLs)'}")
        
        logger.info(f"Workers: {num_workers}, Chunks: {num_chunks}")
        logger.info(f"{'='*80}")
        
        # In gap-fill modus bepalen we de start datum (als niet al opgegeven)
        actual_start = start_date
        if gap_fill and not actual_start:
            actual_start = self.find_first_gap(table_name)
            if actual_start:
                logger.info(f"Vroegste gap gevonden op: {actual_start}")
            else:
                # REASON: Voor consistentie met andere backfill stappen
                actual_start = (datetime.now(timezone.utc) - timedelta(hours=48))
                logger.info(f"✅ Geen gaps gevonden! Geforceerde correctie vanaf {actual_start.strftime('%Y-%m-%d %H:%M')}")
        
        # Check huidige status
        total, null_count, filled_count = self.get_null_count(table)
        logger.info(f"Status: {total:,} total, {filled_count:,} filled, {null_count:,} NULL")
        
        if not gap_fill and null_count == 0:
            logger.info("✅ All records already filled!")
            return {
                'table': table_name,
                'total': total,
                'filled_before': filled_count,
                'filled_after': filled_count,
                'updated': 0,
                'duration': 0
            }
        
        # Get time ranges voor chunks
        time_ranges = self.get_time_ranges(table, num_chunks, start_date=actual_start)
        
        if not time_ranges:
            logger.warning("⚠️ No time ranges found")
            return {
                'table': table_name,
                'total': total,
                'filled_before': filled_count,
                'filled_after': filled_count,
                'updated': 0,
                'duration': 0
            }
        
        logger.info(f"Processing {len(time_ranges)} time chunks...")
        
        # Process chunks in parallel
        start_time = time.time()
        total_fetched = 0
        total_updated = 0
        
        # Progress bar
        pbar = None
        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(
                total=len(time_ranges),
                desc=f"  {table_name:8s}",
                unit="chunk",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] | Updated: {postfix}',
                postfix="0"
            )
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks
            futures = {
                executor.submit(
                    self.process_chunk, table, t_start, t_end, i, gap_fill=gap_fill
                ): i for i, (t_start, t_end) in enumerate(time_ranges)
            }
            
            # Process results
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    fetched, updated, duration = future.result()
                    total_fetched += fetched
                    total_updated += updated
                    
                except Exception as e:
                    logger.error(f"Chunk {chunk_id} exception: {e}")
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"{total_updated:,}")
        
        if pbar:
            pbar.close()
        
        duration = time.time() - start_time
        
        # Final status check
        _, _, filled_after = self.get_null_count(table)
        
        logger.info(f"\n✅ Completed in {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"Fetched: {total_fetched:,}, Updated: {total_updated:,}")
        logger.info(f"Final: {filled_after:,} filled ({100.0 * filled_after / total:.2f}%)")
        
        return {
            'table': table_name,
            'total': total,
            'filled_before': filled_count,
            'filled_after': filled_after,
            'updated': total_updated,
            'duration': duration
        }
    
    def backfill_all(
        self,
        num_workers: int = ATR_MAX_WORKERS,
        num_chunks: int = 100,
        show_progress: bool = True,
        gap_fill: bool = False,
        start_date: Optional[datetime] = None
    ) -> List[Dict]:
        """Backfill alle MTF tabellen."""
        results = []
        
        for table_name in MTF_TABLES:
            result = self.backfill_table(
                table_name, num_workers, num_chunks, show_progress, gap_fill=gap_fill, start_date=start_date
            )
            results.append(result)
        
        return results


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Client-Side ATR Backfill v2 - joins happen on Windows (96GB RAM)'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of parallel workers (default: 8)'
    )
    parser.add_argument(
        '--chunks', type=int, default=200,
        help='Number of time-based chunks (default: 200)'
    )
    parser.add_argument(
        '--table', choices=['lead', 'coin', 'conf', 'all'], default='all',
        help='Table to process (default: all)'
    )
    parser.add_argument(
        '--gap-fill', action='store_true',
        help='Zoek vroegste gap en overschrijf vanaf daar (voor instabiele data)'
    )
    parser.add_argument(
        '--start-date', type=str,
        help='Start datum (YYYY-MM-DD) voor volledige herberekening (overschrijft bestaande data)'
    )
    parser.add_argument(
        '--db', type=str, default=DEFAULT_DB_CONNECTION,
        help='Database connection string'
    )
    
    args = parser.parse_args()
    
    # Parse start_date indien opgegeven
    start_dt = None
    if args.start_date:
        try:
            start_dt = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError as e:
            logger.error(f"Ongeldige start-date format: {e}")
            sys.exit(1)
    
    logger.info("="*80)
    logger.info("ATR BACKFILL v2 - CLIENT-SIDE JOINS")
    if start_dt:
        logger.info(f"Mode: FULL (vanaf {args.start_date})")
    else:
        logger.info(f"Mode: {'GAP-FILL' if args.gap_fill else 'NORMAL'}")
    logger.info("="*80)
    
    backfiller = ATRBackfillClientSide(args.db)
    
    overall_start = time.time()
    
    if args.table == 'all':
        results = backfiller.backfill_all(args.workers, args.chunks, gap_fill=args.gap_fill, start_date=start_dt)
    else:
        results = [backfiller.backfill_table(args.table, args.workers, args.chunks, gap_fill=args.gap_fill, start_date=start_dt)]
    
    overall_duration = time.time() - overall_start
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    total_updated = sum(r['updated'] for r in results)
    
    for result in results:
        pct = 100.0 * result['filled_after'] / result['total'] if result['total'] > 0 else 0
        logger.info(
            f"{result['table']:8s}: {result['updated']:>12,} updated, "
            f"{result['filled_after']:>12,} filled ({pct:.2f}%), "
            f"{result['duration']:>8.1f}s"
        )
    
    logger.info(f"\nTotal: {total_updated:,} records updated")
    logger.info(f"Duration: {overall_duration:.1f}s ({overall_duration/60:.1f} minutes)")
    
    if total_updated > 0 or args.gap_fill:
        logger.info("✅ SUCCESS: ATR backfill v2 completed!")
        return 0
    else:
        logger.warning("⚠️ No records updated. Check if data exists in kfl.indicators")
        return 1


if __name__ == '__main__':
    sys.exit(main())

