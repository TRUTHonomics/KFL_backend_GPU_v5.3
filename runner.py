import sys
import subprocess
import os
import argparse
import psycopg2
from pathlib import Path
from datetime import datetime, timezone, timedelta

# KFL logregels: log file in _log/
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from utils.kfl_logging import setup_kfl_logging

def get_db_connection_string():
    """Genereer database connection string op basis van environment variabelen."""
    user = os.getenv("POSTGRES_USER", "kfl_gpu_backfill")
    password = os.getenv("POSTGRES_PASSWORD", "1234")
    host = os.getenv("POSTGRES_HOST", "10.10.10.3")
    port = os.getenv("POSTGRES_PORT", "5432")
    name = os.getenv("POSTGRES_DB", "kflhyper")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"

def get_selected_asset_ids():
    """Haal asset IDs op waar selected_in_current_run = 1"""
    try:
        conn = psycopg2.connect(get_db_connection_string())
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM symbols.symbols WHERE selected_in_current_run = 1 ORDER BY id")
        asset_ids = [str(row[0]) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return asset_ids
    except Exception as e:
        print(f"⚠️  Kon selected assets niet ophalen: {e}")
        return []

def run_mtf_step(asset_ids, intervals, start_date_str, mode, env):
    """Voer de MTF backfill stap uit."""
    print("\n" + "="*70)
    print("🚀 STAP 2: MTF Backfill (Multi-TimeFrame)")
    print("="*70)
    
    try:
        # Importeer MTF logica vanuit de src/ directory
        sys.path.insert(0, '/app/src')
        from backfill.writers.mtf_backfill import MTFBackfillWriter, create_mtf_indexes
        from backfill.config import MTF_MAX_WORKERS, MTF_CHUNK_DAYS
        
        db_conn = get_db_connection_string()
        create_mtf_indexes(db_conn)
        writer = MTFBackfillWriter(db_conn)
        
        # Filter intervallen die MTF ondersteunt
        mtf_intervals = [i for i in intervals if i in ['1', '60', '240', 'D']]
        if not mtf_intervals:
            print("⚠️  Geen MTF-compatibele intervallen (1, 60, 240, D) geselecteerd.")
            return True

        # Bepaal start/eind datum
        is_gap_fill = (mode == 'gaps_only')
        start_dt = None
        if start_date_str and not is_gap_fill:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        # Eind datum is 'nu'
        end_dt = datetime.now(timezone.utc)
        
        print(f"🔄 MTF Backfill voor {len(asset_ids)} assets...")
        results = writer.backfill_parallel(
            asset_ids=[int(aid) for aid in asset_ids],
            start_date=start_dt,
            end_date=end_dt,
            intervals=mtf_intervals,
            gap_fill=is_gap_fill,
            max_workers=MTF_MAX_WORKERS,
            chunk_days=MTF_CHUNK_DAYS
        )
        
        total_rows = sum(sum(r.values()) for r in results.values() if isinstance(r, dict) and 'error' not in r)
        print(f"✅ MTF voltooid: {total_rows:,} rijen geschreven.")
        return True
        
    except Exception as e:
        print(f"❌ MTF stap gefaald: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger = setup_kfl_logging("runner")
    logger.info("Command: %s", " ".join(sys.argv))

    # Parse arguments passed from backend menu
    parser = argparse.ArgumentParser(description='Compatibility shim for GPU Backfill')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--start-date', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--all-assets', action='store_true')
    parser.add_argument('--use-selected-assets', action='store_true')
    parser.add_argument('--asset-ids', type=str)
    parser.add_argument('--intervals', type=str)
    parser.add_argument('--scripts', type=str)
    
    if '--help' in sys.argv:
        env = os.environ.copy()
        env['PYTHONPATH'] = '/app/src'
        subprocess.run([sys.executable, '-m', 'backfill', '--help'], env=env)
        return

    args, unknown = parser.parse_known_args()
    
    env = os.environ.copy()
    env['PYTHONPATH'] = '/app/src'
    
    # --- STAP 1: INDICATORS & SIGNALS ---
    print("\n" + "="*70)
    print("🚀 STAP 1: Indicators & Signals Backfill")
    print("="*70)
    
    cmd = [sys.executable, '-m', 'backfill']
    
    # Map mode
    backfill_mode = 'gaps_only' if args.mode == 'gaps_only' else 'full'
    cmd.extend(['--mode', backfill_mode])
        
    # Asset selectie
    final_asset_ids = []
    if args.all_assets:
        cmd.append('--all')
        # Voor MTF moeten we weten welke assets dit zijn
        try:
            conn = psycopg2.connect(get_db_connection_string())
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT asset_id FROM kfl.klines_raw")
            final_asset_ids = [str(row[0]) for row in cur.fetchall()]
            conn.close()
        except: pass
    elif args.asset_ids:
        final_asset_ids = args.asset_ids.split(',')
        cmd.extend(['--asset_ids', args.asset_ids])
    elif args.use_selected_assets:
        final_asset_ids = get_selected_asset_ids()
        if final_asset_ids:
            print(f"📊 Gedetecteerd {len(final_asset_ids)} geselecteerde assets voor MTF")
            cmd.extend(['--asset_ids', ','.join(final_asset_ids)])
        else:
            print("⚠️  Geen geselecteerde assets gevonden in symbols.symbols (selected_in_current_run=1)")
            print("💡 Gebruik --all als fallback voor Stap 1")
            cmd.append('--all')
    else:
        cmd.append('--all')
        
    # Intervals
    intervals = args.intervals.split(',') if args.intervals else ['1', '60', '240', 'D']
    if args.intervals:
        cmd.extend(['--intervals', args.intervals])
        
    # Start date
    start_date_str = None
    if args.start_date:
        start_date_str = args.start_date.split('T')[0]
        cmd.extend(['--start_date', start_date_str])
        
    if args.verbose:
        cmd.append('--verbose')
        
    print(f"🏃 Start Backfill CLI: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"❌ Stap 1 (Backfill) gefaald met exit code {result.returncode}. MTF wordt overgeslagen.")
        sys.exit(result.returncode)
        
    print("✅ Stap 1 voltooid.")

    # --- STAP 2: MTF BACKFILL ---
    # REASON: Automatiseer MTF stap zodat backend_v3 Optie 1 de volledige keten uitvoert.
    if not final_asset_ids:
        # Fallback: als we nog steeds geen IDs hebben, haal selected assets op
        final_asset_ids = get_selected_asset_ids()

    if final_asset_ids:
        mtf_success = run_mtf_step(final_asset_ids, intervals, start_date_str, backfill_mode, env)
        if not mtf_success:
            sys.exit(1)
    else:
        print("⚠️  Geen asset_ids bekend voor MTF stap. Skip MTF.")

    print("\n🎉 VOLLEDIGE BACKFILL SEQUENCE VOLTOOID")
    sys.exit(0)

if __name__ == '__main__':
    main()
