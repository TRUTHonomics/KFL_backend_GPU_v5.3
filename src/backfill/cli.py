"""
Command-line interface for indicator and signal backfill.

Usage:
    python -m backfill.cli --asset_id 1 --interval_min 1 --mode full
    python -m backfill.cli --all --mode gaps_only
    python -m backfill.cli --asset_id 1 --start_date 2024-01-01 --end_date 2024-12-31
"""

import argparse
import fcntl
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import DEFAULT_DB_CONNECTION
from .pipeline import BackfillPipeline
from utils.kfl_logging import setup_kfl_logging

# Log directory for lock file (same as KFL _log/)
LOG_DIR = Path("/app/_log") if os.path.exists("/app") else Path(__file__).resolve().parent.parent.parent / "_log"
ARCHIVE_DIR = LOG_DIR / "archive"
LOCK_FILE = LOG_DIR / ".backfill.lock"


class PipelineLock:
    """
    File-based lock to prevent concurrent pipeline runs.
    
    REASON: Voorkomt dat meerdere pipeline instanties tegelijk draaien,
    wat staging table conflicten en data corruptie kan veroorzaken.
    """
    
    def __init__(self, lock_path: Path = LOCK_FILE):
        self.lock_path = lock_path
        self.lock_file = None
        self.locked = False
    
    def acquire(self) -> bool:
        """
        Try to acquire the lock.
        
        Returns:
            True if lock acquired, False if another instance is running.
        """
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_file = open(self.lock_path, 'w')
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Schrijf PID naar lock file voor debugging
            self.lock_file.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
            self.lock_file.flush()
            self.locked = True
            return True
        except (IOError, OSError):
            # Lock is al bezet door andere instantie
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None
            return False
    
    def release(self):
        """Release the lock."""
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
            except Exception:
                pass
            finally:
                self.lock_file = None
                self.locked = False
    
    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(
                "❌ Een andere backfill pipeline draait al!\n"
                "   Wacht tot deze klaar is of stop deze eerst.\n"
                f"   Lock file: {self.lock_path}"
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string to datetime."""
    if not date_str:
        return None
    
    # Try multiple formats
    formats = [
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date: {date_str}")


def parse_list(value: Optional[str]) -> Optional[List]:
    """Parse comma-separated list, stripping quotes."""
    if not value:
        return None
    result = []
    for v in value.split(','):
        v = v.strip()
        # Strip surrounding quotes (single or double)
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            v = v[1:-1]
        result.append(v)
    return result


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated list of integers."""
    if not value:
        return None
    return [int(v.strip()) for v in value.split(',')]


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Backfill indicators and signals for KlineFuturesLab',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full backfill for one asset/interval
  python -m backfill.cli --asset_id 1 --interval_min 1 --mode full

  # Gap-fill for all assets, specific intervals
  python -m backfill.cli --all --intervals 1,5,15 --mode gaps_only

  # Date range backfill
  python -m backfill.cli --asset_id 1 --interval_min D --start_date 2024-01-01 --end_date 2024-12-31

  # Multiple assets
  python -m backfill.cli --asset_ids 1,2,3 --interval_min 1 --mode full
        """
    )
    
    # Asset/interval selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all', 
        action='store_true',
        help='Process all assets and intervals'
    )
    group.add_argument(
        '--asset_id',
        type=int,
        help='Single asset ID to process'
    )
    
    parser.add_argument(
        '--asset_ids',
        type=str,
        help='Comma-separated list of asset IDs'
    )
    
    parser.add_argument(
        '--interval_min',
        type=str,
        help='Single interval to process (e.g., 1, 5, 15, 60, 240, D)'
    )
    
    parser.add_argument(
        '--intervals',
        type=str,
        help='Comma-separated list of intervals'
    )
    
    # Date range
    parser.add_argument(
        '--start_date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'gaps_only', 'incremental'],
        default='full',
        help='Global processing mode (default: full)'
    )
    
    parser.add_argument(
        '--mode-indicators',
        type=str,
        choices=['full', 'gaps_only', 'incremental'],
        help='Override mode for indicators'
    )
    
    parser.add_argument(
        '--mode-signals',
        type=str,
        choices=['full', 'gaps_only', 'incremental'],
        help='Override mode for signals'
    )
    
    # GPU
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    # Database
    parser.add_argument(
        '--db',
        type=str,
        default=DEFAULT_DB_CONNECTION,
        help='Database connection string'
    )
    
    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Only show errors'
    )
    
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO)
    logger = setup_kfl_logging("backfill", log_level=level)
    logger.info("Command: %s", " ".join(sys.argv))
    
    # Parse arguments
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
    except ValueError as e:
        logger.error(f"Date parsing error: {e}")
        sys.exit(1)
    
    # Determine asset_ids
    asset_ids = None
    if args.all:
        asset_ids = None  # Will fetch all
    elif args.asset_ids:
        asset_ids = parse_int_list(args.asset_ids)
    elif args.asset_id:
        asset_ids = [args.asset_id]
    else:
        # Default: require explicit selection
        logger.error("Must specify --all, --asset_id, or --asset_ids")
        sys.exit(1)
    
    # Determine intervals
    intervals = None
    if args.intervals:
        intervals = parse_list(args.intervals)
    elif args.interval_min:
        intervals = [args.interval_min]
    elif not args.all:
        logger.error("Must specify --interval_min or --intervals (or use --all)")
        sys.exit(1)
    
    # REASON: Lock voorkomt dat meerdere pipelines tegelijk draaien
    try:
        with PipelineLock() as lock:
            # Initialize pipeline
            logger.info("Initializing backfill pipeline...")
            pipeline = BackfillPipeline(
                connection_string=args.db,
                use_gpu=not args.no_gpu
            )
            
            # Run pipeline
            mode_ind = args.mode_indicators or args.mode
            mode_sig = args.mode_signals or args.mode
            
            logger.info(f"Starting backfill (indicators={mode_ind}, signals={mode_sig})")
            logger.info(f"Assets: {asset_ids or 'all'}")
            logger.info(f"Intervals: {intervals or 'all'}")
            logger.info(f"Date range: {start_date or 'beginning'} to {end_date or 'now'}")
            
            try:
                results = pipeline.run(
                    asset_ids=asset_ids,
                    intervals=intervals,
                    start_date=start_date,
                    end_date=end_date,
                    mode_indicators=mode_ind,
                    mode_signals=mode_sig
                )
                
                # Summary
                logger.info("=" * 60)
                logger.info("BACKFILL COMPLETE")
                logger.info("=" * 60)
                logger.info(f"Indicators written:  {results['indicators']:,}")
                logger.info(f"Leading signals:     {results['signals_lead']:,}")
                logger.info(f"Coincident signals:  {results['signals_coin']:,}")
                logger.info(f"Confirming signals:  {results['signals_conf']:,}")
                logger.info("=" * 60)
                logger.info(f"Finished: {datetime.now(timezone.utc).isoformat()}")
                
            except KeyboardInterrupt:
                logger.warning("Backfill interrupted by user")
                logger.info(f"Interrupted: {datetime.now(timezone.utc).isoformat()}")
                sys.exit(130)
            except Exception as e:
                logger.error(f"Pipeline failed: {e}", exc_info=True)
                logger.info(f"Failed: {datetime.now(timezone.utc).isoformat()}")
                sys.exit(1)
    
    except RuntimeError as e:
        # Lock kon niet verkregen worden - andere pipeline draait al
        logger.error(str(e))
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
