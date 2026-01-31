#!/usr/bin/env python3
"""
Parallel ATR Backfill Script voor KFL Backend GPU v5.3

Vult atr_at_signal waarden in MTF tabellen vanuit kfl.indicators (niet indicators_unified_cache).
Gebruikt parallelisatie met configureerbaar aantal workers (default: 8) voor snelle verwerking van ~87M records per tabel.

REASON: indicators_unified_cache is een cache tabel met alleen de meest recente kline.
         Historische data staat in kfl.indicators.

INTEGRATIE: Dit script wordt uitgevoerd na MTF backfill om historische ATR waarden te vullen.
            Triggers vullen alleen nieuwe records automatisch.

USAGE:
    # Via docker-menu.py (optie 7)
    # Of direct:
    python -m src.backfill.scripts.atr_backfill --workers 8 --table all
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import OperationalError, InterfaceError

# KFL Backend imports
from ..config import DEFAULT_DB_CONNECTION, ATR_MAX_WORKERS, ATR_BATCH_SIZE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MTF tabellen mapping
MTF_TABLES = {
    'lead': 'kfl.mtf_signals_lead',
    'coin': 'kfl.mtf_signals_coin',
    'conf': 'kfl.mtf_signals_conf'
}


def get_db_connection():
    """Get database connection using KFL backend config"""
    return psycopg2.connect(DEFAULT_DB_CONNECTION)


def get_cursor(commit=False):
    """
    Get database cursor. Returns (conn, cur) if commit=True, else just cur.
    
    REASON: Aangepast voor KFL backend database structuur.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    if commit:
        return conn, cur
    return cur


class ATRBackfillWorker:
    """Worker voor parallel ATR backfill"""
    
    def __init__(self, table: str, batch_size: int = 100000):
        self.table = table
        self.batch_size = batch_size
        self.processed = 0
        self.updated = 0
        
    def get_batch_ranges(self, num_batches: int) -> List[Tuple[Optional[datetime], Optional[datetime]]]:
        """
        Verdeel de tabel in time-based batches voor efficiënte hypertable processing.
        Returns list van (time_start, time_end) tuples.
        """
        query = f"""
        SELECT 
            MIN(time_1) as min_time,
            MAX(time_1) as max_time,
            COUNT(*) as total_count
        FROM {self.table}
        WHERE atr_at_signal IS NULL
        """
        
        try:
            cur = get_cursor()
            cur.execute(query)
            row = cur.fetchone()
            min_time = row[0]
            max_time = row[1]
            total_count = row[2]
            cur.close()
            
            if min_time is None or max_time is None or total_count == 0:
                return []
            
            # Verdeel tijd range in batches
            ranges = []
            time_diff = max_time - min_time
            batch_interval = time_diff / num_batches if num_batches > 0 else time_diff
            
            current_time = min_time
            for i in range(num_batches):
                if i == num_batches - 1:
                    # Laatste batch: tot max_time
                    ranges.append((current_time, max_time))
                else:
                    end_time = current_time + batch_interval
                    ranges.append((current_time, end_time))
                    current_time = end_time + timedelta(seconds=1)  # +1 sec om overlap te voorkomen
                    
            logger.info(f"  {self.table}: {len(ranges)} time-based batches van {min_time} tot {max_time}")
            return ranges
            
        except Exception as e:
            logger.error(f"  Error getting batch ranges for {self.table}: {e}")
            return []
    
    def process_batch(self, time_start: Optional[datetime], time_end: Optional[datetime], worker_id: int, max_retries: int = 3) -> Tuple[int, int]:
        """
        Verwerk één batch van records op basis van time range.
        Returns (processed_count, updated_count)
        
        REASON: Retry logic voor connection errors (database restart/timeout).
        """
        # REASON: Time-based batching werkt beter voor TimescaleDB hypertables
        query = f"""
        UPDATE {self.table} mtf
        SET atr_at_signal = iuc.atr_14
        FROM kfl.indicators iuc
        WHERE mtf.atr_at_signal IS NULL
          AND mtf.time_1 >= %s
          AND mtf.time_1 <= %s
          AND mtf.asset_id = iuc.asset_id
          AND mtf.time_1 = iuc.time
          AND iuc.interval_min = '1'::kfl.interval_type
          AND iuc.atr_14 IS NOT NULL
          AND iuc.atr_14 > 0
        """
        
        for attempt in range(max_retries):
            try:
                conn, cur = get_cursor(commit=True)
                cur.execute(query, (time_start, time_end))
                updated = cur.rowcount
                conn.commit()
                cur.close()
                conn.close()
                
                # Schat processed count (niet exact, maar indicatief)
                processed = updated  # Conservatieve schatting
                return processed, updated
                
            except (OperationalError, InterfaceError) as e:
                error_msg = str(e)
                is_connection_error = (
                    'EOF detected' in error_msg or
                    'connection' in error_msg.lower() or
                    'shutting down' in error_msg.lower() or
                    'server closed' in error_msg.lower()
                )
                
                if is_connection_error and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconden
                    logger.warning(f"  Worker {worker_id}: Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"  Worker {worker_id}: Error processing batch {time_start} to {time_end}: {e}")
                    return 0, 0
                    
            except Exception as e:
                logger.error(f"  Worker {worker_id}: Error processing batch {time_start} to {time_end}: {e}")
                return 0, 0
        
        # Alle retries gefaald
        logger.error(f"  Worker {worker_id}: Failed after {max_retries} attempts for batch {time_start} to {time_end}")
        return 0, 0


def backfill_table(table_name: str, table_schema: str, num_workers: int = ATR_MAX_WORKERS, batch_size: int = ATR_BATCH_SIZE) -> dict:
    """
    Backfill ATR voor één MTF tabel met parallelisatie.
    
    Args:
        table_name: Tabel naam (lead/coin/conf)
        table_schema: Volledige schema.tabel naam
        num_workers: Aantal parallel workers
        batch_size: Batch grootte
        
    Returns:
        Dict met statistieken
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting ATR backfill for {table_schema}")
    logger.info(f"Workers: {num_workers}, Batch size: {batch_size}")
    logger.info(f"{'='*80}")
    
    # Check huidige status
    cur = get_cursor()
    cur.execute(f"""
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE atr_at_signal IS NULL) as null_count,
            COUNT(*) FILTER (WHERE atr_at_signal IS NOT NULL) as filled_count
        FROM {table_schema}
    """)
    row = cur.fetchone()
    total = row[0]
    null_count = row[1]
    filled_count = row[2]
    cur.close()
        
    logger.info(f"  Status: {total:,} total, {filled_count:,} filled, {null_count:,} NULL")
    
    if null_count == 0:
        logger.info(f"  ✅ All records already filled!")
        return {
            'table': table_name,
            'total': total,
            'filled_before': filled_count,
            'filled_after': filled_count,
            'updated': 0,
            'duration': 0
        }
    
    # Maak worker
    worker = ATRBackfillWorker(table_schema, batch_size)
    
    # Bepaal aantal batches op basis van aantal workers
    # REASON: Meer batches = betere load balancing en minder geheugengebruik per transactie
    # Verhoogd van 100 naar 2000 voor grote datasets (~50K records per batch bij 100M records)
    num_batches = max(num_workers * 20, 2000) 
    
    # Get batch ranges (time-based voor hypertables)
    batch_ranges = worker.get_batch_ranges(num_batches)
    
    if not batch_ranges:
        logger.warning(f"  ⚠️  No batches to process")
        return {
            'table': table_name,
            'total': total,
            'filled_before': filled_count,
            'filled_after': filled_count,
            'updated': 0,
            'duration': 0
        }
    
    # Process batches in parallel
    start_time = time.time()
    total_processed = 0
    total_updated = 0
    
    # Initialize progress bar
    if TQDM_AVAILABLE:
        pbar = tqdm(
            total=len(batch_ranges),
            desc=f"  {table_name:8s}",
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] | Updated: {postfix}',
            postfix="0"
        )
    else:
        pbar = None
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(worker.process_batch, time_start, time_end, i): (time_start, time_end, i)
            for i, (time_start, time_end) in enumerate(batch_ranges)
        }
        
        # Process completed batches
        completed = 0
        for future in as_completed(future_to_batch):
            time_start, time_end, batch_id = future_to_batch[future]
            try:
                processed, updated = future.result()
                total_processed += processed
                total_updated += updated
                completed += 1
                
                # Update progress bar
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(f"{total_updated:,}")
                elif completed % 10 == 0 or completed == len(batch_ranges):
                    logger.info(f"  Progress: {completed}/{len(batch_ranges)} batches, "
                              f"{total_updated:,} records updated")
                              
            except Exception as e:
                logger.error(f"  Batch {batch_id} ({time_start} to {time_end}) failed: {e}")
                if pbar:
                    pbar.update(1)
    
    # Close progress bar
    if pbar:
        pbar.close()
    
    duration = time.time() - start_time
    
    # Check final status
    cur = get_cursor()
    cur.execute(f"""
        SELECT COUNT(*) FILTER (WHERE atr_at_signal IS NOT NULL) as filled_count
        FROM {table_schema}
    """)
    filled_after = cur.fetchone()[0]
    cur.close()
    
    logger.info(f"\n  ✅ Completed in {duration:.1f}s")
    logger.info(f"  Updated: {total_updated:,} records")
    logger.info(f"  Final status: {filled_after:,} filled ({100.0 * filled_after / total:.2f}%)")
    
    return {
        'table': table_name,
        'total': total,
        'filled_before': filled_count,
        'filled_after': filled_after,
        'updated': total_updated,
        'duration': duration
    }


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Parallel ATR backfill for MTF tables')
    parser.add_argument('--workers', type=int, default=ATR_MAX_WORKERS, help=f'Number of parallel workers (default: {ATR_MAX_WORKERS} from config)')
    parser.add_argument('--batch-size', type=int, default=ATR_BATCH_SIZE, help=f'Batch size (default: {ATR_BATCH_SIZE} from config)')
    parser.add_argument('--table', choices=['lead', 'coin', 'conf', 'all'], default='all',
                       help='Table to process (default: all)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("PARALLEL ATR BACKFILL")
    logger.info("="*80)
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Tables: {args.table}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine tables to process
    if args.table == 'all':
        tables_to_process = list(MTF_TABLES.items())
    else:
        tables_to_process = [(args.table, MTF_TABLES[args.table])]
    
    # Process each table
    results = []
    overall_start = time.time()
    
    for table_name, table_schema in tables_to_process:
        # REASON: Kleinere batches en minder workers voorkomen OOM op de DB server
        result = backfill_table(table_name, table_schema, args.workers, args.batch_size)
        results.append(result)
    
    overall_duration = time.time() - overall_start
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    total_updated = sum(r['updated'] for r in results)
    total_filled_before = sum(r['filled_before'] for r in results)
    total_filled_after = sum(r['filled_after'] for r in results)
    
    for result in results:
        logger.info(f"{result['table']:8s}: "
                   f"{result['updated']:>12,} updated, "
                   f"{result['filled_after']:>12,} filled "
                   f"({100.0 * result['filled_after'] / result['total']:.2f}%), "
                   f"{result['duration']:>8.1f}s")
    
    logger.info(f"\nTotal: {total_updated:,} records updated")
    logger.info(f"Duration: {overall_duration:.1f}s ({overall_duration/60:.1f} minutes)")
    logger.info(f"Final filled: {total_filled_after:,} records")
    
    if total_updated > 0:
        logger.info(f"✅ SUCCESS: ATR backfill completed!")
    else:
        logger.warning(f"⚠️  No records were updated. Check if data exists in kfl.indicators")
    
    return 0 if total_updated > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
