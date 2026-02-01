#!/usr/bin/env python3
"""
Docker Menu Script voor KFL Backend GPU v5.3
===========================================

Interactief menu om de GPU backfill pipeline te starten met configureerbare opties.
Draai dit script IN de Docker container via: docker exec -it KFL_backend_GPU_v5_3 python /app/docker-menu.py
Of lokaal op Windows (zonder file locking).
"""

import os
import sys
import subprocess
import time
import yaml
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import psycopg2

# REASON: Cross-platform file locking - fcntl werkt alleen op Linux
IS_WINDOWS = os.name == 'nt'
if not IS_WINDOWS:
    import fcntl

# Project root
PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOG_DIR = PROJECT_ROOT / "_log"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
from utils.kfl_logging import setup_kfl_logging


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def load_config() -> Dict[str, Any]:
    """Laad config.yaml"""
    try:
        if not CONFIG_PATH.exists():
            return {}
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️  Kon config niet laden: {e}")
        return {}


def save_config(config: Dict[str, Any]) -> bool:
    """Sla config.yaml op"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"❌ Fout bij opslaan config: {e}")
        return False


def get_db_connection_string(config: Dict[str, Any]) -> str:
    """
    Genereer database connection string.
    REASON: Gebruik environment variabelen uit .env.local als Single Source of Truth.
    """
    db_user = os.getenv("POSTGRES_USER") or os.getenv("DB_USER")
    db_pass = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASS")
    db_host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST")
    db_port = os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT")
    db_name = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME")

    db_cfg = config.get('database', {})
    user = db_user or db_cfg.get('user', 'kfl_gpu_backfill')
    password = db_pass or db_cfg.get('password', '1234')
    host = db_host or db_cfg.get('host', '10.10.10.3')
    port = db_port or db_cfg.get('port', 5432)
    name = db_name or db_cfg.get('name', 'kflhyper')

    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def test_database_connection(config: Dict[str, Any]) -> bool:
    """Test database connectie."""
    try:
        db_conn_string = get_db_connection_string(config)
        conn = psycopg2.connect(db_conn_string, connect_timeout=10)
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database connectie gefaald: {e}")
        return False


def get_selected_asset_ids(config: Dict[str, Any]) -> List[int]:
    """Haal asset IDs op waar selected_in_current_run = 1"""
    try:
        db_conn_string = get_db_connection_string(config)
        conn = psycopg2.connect(db_conn_string)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM symbols.symbols 
            WHERE selected_in_current_run = 1
            ORDER BY id
        """)
        asset_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return asset_ids
    except Exception as e:
        print(f"⚠️  Kon selected assets niet ophalen: {e}")
        return []


def archive_old_logs(pattern: str = "*.log"):
    """Archiveer specifieke oude log bestanden naar de archive map."""
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return

    log_files = [f for f in LOG_DIR.glob(pattern) if f.is_file()]
    if not log_files:
        return

    archive_dir = LOG_DIR / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_subdir = archive_dir / timestamp
    archive_subdir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    print(f"📦 Archiveren van {len(log_files)} logs ({pattern}) naar {archive_subdir}...")
    for log_file in log_files:
        try:
            shutil.move(str(log_file), str(archive_subdir / log_file.name))
        except Exception:
            pass


def check_pipeline_running() -> bool:
    """
    Check of er al een pipeline draait en ruim stale locks op.
    REASON: Cross-platform - op Windows skip file locking, op Linux gebruik fcntl.
    """
    lock_file = LOG_DIR / '.backfill.lock'
    if not lock_file.exists():
        return False

    # Op Windows: simpele check op bestandsinhoud (geen echte locking)
    if IS_WINDOWS:
        try:
            with open(lock_file, 'r') as f:
                lines = f.readlines()
            if lines:
                # Check of PID nog actief is (Windows)
                try:
                    pid = int(lines[0].strip())
                    # tasklist check op Windows
                    result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                          capture_output=True, text=True, timeout=5)
                    if str(pid) not in result.stdout:
                        print("🧹 Stale lock file gevonden (proces niet actief), wordt verwijderd...")
                        lock_file.unlink()
                        return False
                except (ValueError, subprocess.TimeoutExpired):
                    pass
                print(f"⚠️  Er draait mogelijk al een pipeline (PID: {lines[0].strip()})")
                if len(lines) > 1:
                    print(f"   Gestart: {lines[1].strip()}")
            return True
        except Exception as e:
            print(f"⚠️  Fout bij checken lock: {e}")
            return False

    # Op Linux: gebruik fcntl voor echte file locking
    try:
        f = open(lock_file, 'r')
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()
            print("🧹 Stale lock file gevonden, wordt verwijderd...")
            lock_file.unlink()
            return False
        except (IOError, OSError):
            f.seek(0)
            lines = f.readlines()
            f.close()
            if lines:
                print(f"⚠️  Er draait al een pipeline (PID: {lines[0].strip()})")
                if len(lines) > 1:
                    print(f"   Gestart: {lines[1].strip()}")
            return True
    except Exception as e:
        print(f"⚠️  Fout bij checken lock: {e}")
        return False


def build_command(proc_config: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    """Bouw command line voor backfill script"""
    cmd = [sys.executable, '-m', 'backfill.cli']
    db_conn = get_db_connection_string(config)
    cmd.extend(['--db', db_conn])
    if proc_config.get('start_date'):
        cmd.extend(['--start_date', proc_config['start_date']])
    if proc_config.get('end_date'):
        cmd.extend(['--end_date', proc_config['end_date']])
    cmd.extend(['--mode', proc_config.get('mode', 'full')])
    if proc_config.get('asset_selection') == 'all':
        cmd.append('--all')
    elif proc_config.get('asset_ids'):
        cmd.extend(['--asset_ids', ','.join(str(x) for x in proc_config['asset_ids'])])
    if proc_config.get('intervals'):
        cmd.extend(['--intervals', ','.join(str(x) for x in proc_config['intervals'])])
    if not proc_config.get('use_gpu', True):
        cmd.append('--no-gpu')
    if proc_config.get('verbose', False):
        cmd.append('--verbose')
    return cmd


def run_full_backfill_sequence():
    """Menu optie: Volledige sequentiële backfill (Indicators -> Signals -> MTF)"""
    config = load_config()
    print("\n" + "=" * 70)
    print("🔄 VOLLEDIGE SEQUENTIËLE BACKFILL")
    print("=" * 70)
    print("Dit voert achter elkaar uit:")
    print("  1. Indicators & Signals backfill")
    print("  2. MTF backfill")
    print("=" * 70)
    
    if check_pipeline_running():
        print("\n❌ Er draait al een pipeline!")
        input("\nDruk op Enter om terug te gaan...")
        return

    # === MODUS SELECTIE PER STAP ===
    print("\n--- Modus Selectie per stap ---")
    
    # Stap 1: Indicators & Signals
    print("\nStap 1: Indicators & Signals")
    print("  1. full       - Volledige herberekening")
    print("  2. gaps_only  - Alleen ontbrekende data aanvullen")
    choice1 = input("Keuze [1-2, default 2]: ").strip() or '2'
    mode_step1 = 'gaps_only' if choice1 == '2' else 'full'
    
    # Stap 2: MTF
    print("\nStap 2: MTF (Multi-TimeFrame)")
    print("  1. full       - Volledige scan vanaf start datum")
    print("  2. gap_fill   - Automatisch vroegste gap detecteren")
    choice2 = input("Keuze [1-2, default 2]: ").strip() or '2'
    mode_step2 = 'gap_fill' if choice2 == '2' else 'full'
    
    # REASON: ATR/Outcome backfill is verplaatst naar QBN_v2 container
    # De GPU backfill pipeline doet nu alleen indicators, signals en MTF
    print("\n⚠️  Let op: ATR/Outcome backfill is verplaatst naar QBN_v2 container")
    print("   Draai na deze pipeline: QBN_v2 > Outcome Backfill (optie a)")
    
    # === ASSET SELECTIE ===
    print("\n--- Asset Selectie ---")
    print("  1. Alle assets uit database")
    print("  2. Alleen actieve assets (selected_in_current_run = 1)")
    print("  3. Handmatige selectie uit lijst")
    
    asset_ids = []
    asset_choice = input("\nKeuze [1-3, default 2]: ").strip()
    asset_selection = 'all'  # Voor CLI flag
    if asset_choice == '1':
        assets = get_assets_with_info(config)
        asset_ids = [a['id'] for a in assets]
        asset_selection = 'all'
    elif asset_choice == '3':
        asset_ids = select_assets_interactively(config)
        asset_selection = 'specific'
    else:
        asset_ids = get_selected_asset_ids(config)
        asset_selection = 'specific'
        
    if not asset_ids and asset_selection != 'all':
        print("⚠️  Geen assets geselecteerd!")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    # === INTERVAL SELECTIE ===
    available_intervals = ['1', '5', '15', '60', '240', 'D']
    default_intervals = ['1', '60', '240', 'D']
    print(f"\n--- Interval Selectie ---")
    print(f"Beschikbare intervallen: {', '.join(available_intervals)}")
    interval_input = input(f"Selecteer intervallen (komma gescheiden) [{','.join(default_intervals)}]: ").strip()
    
    if interval_input:
        intervals = [x.strip() for x in interval_input.split(',') if x.strip() in available_intervals]
        if not intervals:
            print("⚠️  Geen geldige intervallen. Gebruik defaults.")
            intervals = default_intervals
    else:
        intervals = default_intervals
    
    # === DATUM SELECTIE ===
    default_start = '2020-01-01'
    default_end = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    print("\n--- Tijdsbestek ---")
    start_input = input(f"Start datum [{default_start}]: ").strip()
    start_date_str = start_input if start_input else default_start
    
    end_input = input(f"Eind datum [{default_end}]: ").strip()
    end_date_str = end_input if end_input else default_end
    
    try:
        datetime.strptime(start_date_str, '%Y-%m-%d')
        datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError as e:
        print(f"❌ Ongeldige datum format: {e}")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    # === SAMENVATTING ===
    print("\n" + "=" * 70)
    print("📋 CONFIGURATIE SAMENVATTING")
    print("=" * 70)
    print(f"   Stap 1 (Ind/Sig): {mode_step1}")
    print(f"   Stap 2 (MTF):     {mode_step2}")
    print(f"   Assets:           {len(asset_ids) if asset_ids else 'ALLE'} stuks")
    print(f"   Intervallen:      {', '.join(intervals)}")
    print(f"   Periode:          {start_date_str} tot {end_date_str}")
    print("=" * 70)
    
    confirm = input("\nDoorgaan met backfill sequence? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Geannuleerd.")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    log = setup_kfl_logging("full_backfill", project_root=PROJECT_ROOT, log_level=logging.DEBUG)
    log.info("START VOLLEDIGE BACKFILL SEQUENCE")
    log.info("Modi: Step1=%s, Step2=%s", mode_step1, mode_step2)
    log.info("Assets: %s, Intervallen: %s", len(asset_ids) if asset_ids else "ALL", intervals)
    log.info("Periode: %s tot %s", start_date_str, end_date_str)

    overall_start = time.time()

    # === STAP 1: Indicators & Signals Backfill ===
    print("\n" + "=" * 70)
    print(f"STAP 1/2: Indicators & Signals Backfill ({mode_step1})")
    print("=" * 70)
    log.info("STAP 1: Indicators & Signals backfill (mode=%s)", mode_step1)

    proc_config = {
        "mode": mode_step1,
        "asset_selection": asset_selection,
        "asset_ids": asset_ids if asset_selection == "specific" else None,
        "intervals": intervals,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "use_gpu": True,
        "verbose": False,
    }

    if not run_pipeline(proc_config, config):
        log.error("Indicators/Signals backfill gefaald")
        print("\n❌ Indicators/Signals backfill gefaald - sequence afgebroken")
        input("\nDruk op Enter om terug te gaan...")
        return

    print("\n✅ Stap 1 voltooid")

    # === STAP 2: MTF Backfill ===
    print("\n" + "=" * 70)
    print(f"STAP 2/2: MTF Backfill ({mode_step2})")
    print("=" * 70)
    log.info("STAP 2: MTF Backfill (mode=%s)", mode_step2)

    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    mtf_intervals = [i for i in intervals if i in ["1", "60", "240", "D"]]

    try:
        from backfill.writers.mtf_backfill import MTFBackfillWriter, create_mtf_indexes

        db_conn_string = get_db_connection_string(config)
        mtf_asset_ids = asset_ids if asset_ids else get_selected_asset_ids(config)

        if not mtf_asset_ids:
            log.warning("Geen assets voor MTF")
            print("⚠️  Geen assets - skip MTF backfill")
        elif not mtf_intervals:
            log.warning("Geen MTF-compatibele intervallen geselecteerd")
            print("⚠️  Geen MTF intervals (1,60,240,D) - skip MTF backfill")
        else:
            create_mtf_indexes(db_conn_string)
            writer = MTFBackfillWriter(db_conn_string)
            is_gap_fill_mtf = (mode_step2 == "gap_fill")
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt_base = datetime.strptime(end_date_str, "%Y-%m-%d")
            end_dt = (end_dt_base + timedelta(days=1)).replace(tzinfo=timezone.utc)
            log.info("MTF backfill gestart (mode=%s)", mode_step2)
            from backfill.config import MTF_MAX_WORKERS, MTF_CHUNK_DAYS

            results = writer.backfill_parallel(
                asset_ids=mtf_asset_ids,
                start_date=start_dt if not is_gap_fill_mtf else None,
                end_date=end_dt,
                intervals=mtf_intervals,
                gap_fill=is_gap_fill_mtf,
                max_workers=MTF_MAX_WORKERS,
                chunk_days=MTF_CHUNK_DAYS,
            )
            total_rows = sum(sum(r.values()) for r in results.values() if isinstance(r, dict) and "error" not in r)
            log.info("MTF voltooid: %s rijen", f"{total_rows:,}")
            print(f"\n✅ Stap 2 voltooid: {total_rows:,} rijen")

    except Exception as e:
        log.error("MTF backfill gefaald: %s", e, exc_info=True)
        print(f"\n❌ MTF backfill gefaald: {e}")
        input("\nDruk op Enter om terug te gaan...")
        return

    overall_duration = time.time() - overall_start
    print("\n" + "=" * 70)
    print("🎉 GPU BACKFILL SEQUENCE VOLTOOID")
    print("=" * 70)
    print(f"Totale tijd: {overall_duration/60:.1f} minuten")
    print("\n📋 VOLGENDE STAP:")
    print("   Draai QBN_v2 container > Outcome Backfill (optie a)")
    log.info("SEQUENCE VOLTOOID in %.1f minuten", overall_duration / 60)
    for h in log.handlers:
        h.flush()

    input("\nDruk op Enter om terug te gaan...")


def run_pipeline(proc_config: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Voer de backfill pipeline uit"""
    if check_pipeline_running():
        print("\n❌ Kan pipeline niet starten - er draait al een andere instantie!")
        return False
    
    print("\n" + "=" * 70)
    print("🚀 START: Backfill Pipeline")
    print("=" * 70)
    
    cmd = build_command(proc_config, config)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"backfill_{timestamp}.log"
    
    start_time = time.time()
    try:
        with open(log_file, 'w', encoding='utf-8') as log_f:
            log_f.write(f"=== Backfill Pipeline ===\n")
            log_f.write(f"Started: {datetime.now(timezone.utc).isoformat()}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write("=" * 70 + "\n\n")
            log_f.flush()
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(SRC_DIR), bufsize=1, universal_newlines=True)
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
                    log_f.write(line)
                    log_f.flush()
            return_code = process.wait()
        
        elapsed = time.time() - start_time
        if return_code == 0:
            print(f"\n✅ Pipeline voltooid in {elapsed/60:.1f} minuten")
            return True
        else:
            print(f"\n❌ Pipeline gefaald (exit code: {return_code})")
            return False
    except Exception as e:
        print(f"\n❌ Fout bij uitvoeren pipeline: {e}")
        return False


def get_assets_with_info(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Haal alle assets op met ID, naam en active status"""
    try:
        db_conn_string = get_db_connection_string(config)
        conn = psycopg2.connect(db_conn_string)
        cursor = conn.cursor()
        # EXPL: Kolommen heten kraken_symbol/bybit_symbol, niet 'symbol'
        # pair is de leesbare naam (bijv. BTC:USD)
        cursor.execute("""
            SELECT id, COALESCE(pair, kraken_symbol, bybit_symbol) as display_name, selected_in_current_run 
            FROM symbols.symbols 
            ORDER BY id
        """)
        assets = []
        for row in cursor.fetchall():
            assets.append({
                'id': row[0],
                'symbol': row[1],
                'active': bool(row[2])
            })
        cursor.close()
        conn.close()
        return assets
    except Exception as e:
        print(f"⚠️  Kon asset informatie niet ophalen: {e}")
        return []


def select_assets_interactively(config: Dict[str, Any]) -> List[int]:
    """Interactieve selectie van assets uit een lijst"""
    assets = get_assets_with_info(config)
    if not assets:
        return []

    selected_indices = [i for i, a in enumerate(assets) if a['active']]
    
    while True:
        clear_screen()
        print("=" * 70)
        print("     ASSET SELECTIE (Toggle met nummer, 'd' voor klaar)")
        print("=" * 70)
        
        # Toon assets in kolommen voor compactheid
        cols = 3
        rows = (len(assets) + cols - 1) // cols
        
        for r in range(rows):
            line = ""
            for c in range(cols):
                idx = r + c * rows
                if idx < len(assets):
                    asset = assets[idx]
                    status = "✅" if idx in selected_indices else "❌"
                    entry = f"{idx+1:>3}. [{status}] {asset['id']:>4}: {asset['symbol']:<10}"
                    line += entry + "  "
            print(line)
        
        print("\n" + "-" * 70)
        print(f"Geselecteerd: {len(selected_indices)} assets")
        choice = input("\nNummer om te toggelen, 'all' voor alles, 'none' voor niets, of 'd' voor Doorgaan: ").strip().lower()
        
        if choice == 'd':
            break
        elif choice == 'all':
            selected_indices = list(range(len(assets)))
        elif choice == 'none':
            selected_indices = []
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(assets):
                if idx in selected_indices:
                    selected_indices.remove(idx)
                else:
                    selected_indices.append(idx)
    
    return [assets[i]['id'] for i in selected_indices]


def run_atr_backfill():
    """DEPRECATED: ATR/Outcome backfill is verplaatst naar QBN_v2 container"""
    print("\n" + "=" * 70)
    print("⚠️  DEPRECATED: ATR Backfill")
    print("=" * 70)
    print("\nDeze functie is verplaatst naar de QBN_v2 container.")
    print("Outcomes (incl. ATR) worden nu centraal opgeslagen in qbn.signal_outcomes.")
    print("\nGebruik in plaats hiervan:")
    print("  QBN_v2 container > docker-menu.py > optie a (Outcome Backfill)")
    input("\nDruk op Enter om terug te gaan...")
    return
    
    # Oude code hieronder - wordt niet meer uitgevoerd
    print("\n" + "=" * 70)
    print("🚀 START: ATR Backfill v2 (Client-Side Joins)")
    print("=" * 70)
    
    print("\nModus Selectie:")
    print("  1. Normaal    - Alleen ontbrekende NULL waarden vullen")
    print("  2. Gap-fill   - Vanaf vroegste gap alles overschrijven (instabiele data corrigeren)")
    print("  3. Full       - Volledige herberekening vanaf start datum")
    
    choice = input("\nKeuze [1-3, default 1]: ").strip() or '1'
    
    start_dt_str = None
    gap_fill_flag = False
    
    if choice == '3':
        start_dt_str = input(f"\nVoer startdatum in (YYYY-MM-DD) [default 2020-01-01]: ").strip() or "2020-01-01"
        mode_label = f"FULL (vanaf {start_dt_str})"
    elif choice == '2':
        gap_fill_flag = True
        mode_label = "GAP-FILL (overschrijft instabiele data)"
    else:
        mode_label = "NORMAAL (vult alleen NULLs)"
    
    print("\n" + "=" * 70)
    print(f"Modus: {mode_label}")
    print("REASON: Joins gebeuren op Windows (96GB RAM), niet op DB server (48GB)")
    print("        Dit voorkomt OOM crashes bij grote hypertable JOINs")
    print("=" * 70)
    
    # REASON: v2 doet joins lokaal op Windows machine, DB server doet alleen SELECT/UPDATE
    cmd = [sys.executable, '-m', 'backfill.scripts.atr_backfill_v2', '--workers', '8', '--chunks', '200', '--table', 'all']
    if start_dt_str:
        cmd.extend(['--start-date', start_dt_str])
    elif gap_fill_flag:
        cmd.append('--gap-fill')
    
    print(f"Uitvoeren: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(SRC_DIR), bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.rstrip())
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n✅ ATR Backfill v2 voltooid.")
        else:
            print(f"\n❌ ATR Backfill v2 gefaald (exit code: {return_code})")
    except Exception as e:
        print(f"\n❌ Fout bij uitvoeren ATR backfill v2: {e}")
    
    input("\nDruk op Enter om terug te gaan...")


def run_mtf_backfill():
    """Menu optie: MTF (Multi-TimeFrame) backfill met interactieve parameter selectie."""
    config = load_config()
    print("\n" + "=" * 70)
    print("🔄 MTF BACKFILL (Multi-TimeFrame Signals)")
    print("=" * 70)
    
    if check_pipeline_running():
        print("\n❌ Kan MTF backfill niet starten - er draait al een proces!")
        input("\nDruk op Enter om terug te gaan...")
        return

    # === ASSET SELECTIE ===
    print("\nAsset Selectie:")
    print("  1. Alle assets uit database")
    print("  2. Alleen actieve assets (selected_in_current_run = 1)")
    print("  3. Handmatige selectie uit lijst")
    
    asset_ids = []
    asset_choice = input("\nKeuze [1-3, default 2]: ").strip()
    if asset_choice == '1':
        assets = get_assets_with_info(config)
        asset_ids = [a['id'] for a in assets]
    elif asset_choice == '3':
        asset_ids = select_assets_interactively(config)
    else:
        asset_ids = get_selected_asset_ids(config)
        
    if not asset_ids:
        print("⚠️  Geen assets geselecteerd!")
        input("\nDruk op Enter om terug te gaan...")
        return

    # === INTERVAL SELECTIE ===
    # REASON: De MTF tabel heeft vaste kolommen voor d, 240, 60, 1. 
    # Selectie bepaalt welke joins worden uitgevoerd.
    available_intervals = ['1', '60', '240', 'D']
    print(f"\nBeschikbare MTF intervallen: {', '.join(available_intervals)}")
    interval_input = input(f"Selecteer intervallen (komma gescheiden) [{','.join(available_intervals)}]: ").strip()
    
    if interval_input:
        intervals = [x.strip() for x in interval_input.split(',') if x.strip() in available_intervals]
        if not intervals:
            print("⚠️  Geen geldige intervallen geselecteerd. Gebruik defaults.")
            intervals = available_intervals
    else:
        intervals = available_intervals

    # === DATUM SELECTIE ===
    proc_config = config.get('processing', {})
    default_start = '2020-01-01'
    default_end = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    print("\n--- Tijdsbestek ---")
    print("  1. Automatisch vroegste kline detecteren (Full scan)")
    print("  2. Automatische Gap-fill (zoekt vroegste ontbrekende MTF record)")
    print("  3. Handmatige datum invoer")
    
    time_choice = input("\nKeuze [1-3, default 2]: ").strip() or '2'
    
    start_date = None
    start_date_str = "Auto (Full)"
    
    is_gap_fill = (time_choice == '2')
    if is_gap_fill:
        start_date = None
        start_date_str = "Auto (Gap-fill per asset)"
    elif time_choice == '3':
        start_input = input(f"Start datum [{default_start}]: ").strip()
        start_date_str = start_input if start_input else default_start
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError as e:
            print(f"❌ Ongeldige datum format: {e}")
            input("\nDruk op Enter om terug te gaan...")
            return
    elif time_choice == '1':
        start_date = None
        start_date_str = "Auto (Full scan vanaf eerste kline)"
    
    # === CHUNKING & WORKERS ===
    from src.backfill.config import MTF_MAX_WORKERS, MTF_CHUNK_DAYS
    
    print("\n--- Verwerkingsinstellingen ---")
    chunk_input = input(f"Chunk grootte [default={MTF_CHUNK_DAYS}d]: ").strip().lower() or f"{MTF_CHUNK_DAYS}d"
    
    # REASON: Default uit config is nu RAM-vriendelijker
    chunk_days = MTF_CHUNK_DAYS
    chunk_months = 0
    if chunk_input.endswith('d'):
        try:
            chunk_days = int(chunk_input[:-1])
        except ValueError:
            chunk_days = MTF_CHUNK_DAYS
    elif chunk_input.isdigit():
        chunk_months = int(chunk_input)
        chunk_days = 0
    
    # Workers
    workers_input = input(f"\nAantal parallelle workers [{MTF_MAX_WORKERS}]: ").strip()
    max_workers = int(workers_input) if workers_input.isdigit() else MTF_MAX_WORKERS
    
    # Eind datum
    end_input = input(f"Eind datum [{default_end}]: ").strip()
    end_date_str = end_input if end_input else default_end
    try:
        # REASON: end_date 2025-12-25 moet betekenen TOT EN MET die dag (dus 2025-12-26 00:00:00)
        end_dt_base = datetime.strptime(end_date_str, '%Y-%m-%d')
        end_date = (end_dt_base + timedelta(days=1)).replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"❌ Ongeldige datum format: {e}")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    # REASON: Dynamische chunk beschrijving
    if chunk_days > 0:
        chunk_desc = f"{chunk_days} dagen chunks"
    elif chunk_months == 0:
        chunk_desc = "⚡ FAST (geen chunking)"
    else:
        chunk_desc = f"{chunk_months} maanden chunks"
    
    print(f"\n📋 Configuratie:")
    print(f"   Assets:      {len(asset_ids)} stuks")
    print(f"   Intervallen: {', '.join(intervals)}")
    print(f"   Start:       {start_date_str}")
    print(f"   Eind:        {end_date_str}")
    print(f"   Mode:        {chunk_desc}")
    print(f"   Workers:     {max_workers} parallel")
    
    confirm = input("\nDoorgaan met MTF Backfill? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Geannuleerd.")
        input("\nDruk op Enter om terug te gaan...")
        return

    log = setup_kfl_logging("mtf_backfill", project_root=PROJECT_ROOT, log_level=logging.DEBUG)
    log.info("START MTF BACKFILL")
    log.info("Assets: %s, Intervallen: %s", len(asset_ids), intervals)
    log.info("Parameters: %s tot %s, chunk_days=%s, chunk_months=%s", start_date_str, end_date_str, chunk_days, chunk_months)
    
    # === IMPORTS ===
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from backfill.writers.mtf_backfill import MTFBackfillWriter, create_mtf_indexes
        log.info("MTF module geladen")
    except ImportError as e:
        log.error("Import fout: %s", e, exc_info=True)
        print(f"❌ Kon MTF module niet laden: {e}")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    # === DATABASE ===
    db_conn_string = get_db_connection_string(config)
    db_pass = str(config.get('database', {}).get('password', '1234'))
    log.debug("DB connection: %s", db_conn_string.replace(db_pass, "***"))

    if not test_database_connection(config):
        log.error("Database connectie gefaald")
        input("\nDruk op Enter om terug te gaan...")
        return
    
    # === UITVOEREN ===
    print("\n🚀 START: MTF Backfill...")
    try:
        log.info("Aanmaken MTF indexes...")
        for h in log.handlers:
            h.flush()
        create_mtf_indexes(db_conn_string)
        log.info("MTF indexes aangemaakt, initialiseren writer...")
        writer = MTFBackfillWriter(db_conn_string)
        log.info("Writer geïnitialiseerd, start backfill_parallel voor %s assets...", len(asset_ids))
        for h in log.handlers:
            h.flush()
        
        results = writer.backfill_parallel(
            asset_ids=asset_ids, 
            start_date=start_date,  # Kan None zijn voor auto-detect
            end_date=end_date, 
            intervals=intervals,
            max_workers=max_workers,
            chunk_months=chunk_months,
            chunk_days=chunk_days,
            gap_fill=is_gap_fill
        )
        
        total_rows = sum(sum(r.values()) for r in results.values() if isinstance(r, dict) and "error" not in r)
        log.info("Voltooid: %s rijen geschreven", f"{total_rows:,}")
        print(f"\n✅ Voltooid: {total_rows:,} rijen geschreven.")
    except Exception as e:
        log.error("MTF Backfill gefaald: %s", e, exc_info=True)
        print(f"\n❌ Fout: {e}")
        import traceback
        traceback.print_exc()
    
    for h in log.handlers:
        h.flush()
    input("\nDruk op Enter om terug te gaan...")


def show_menu():
    """Toon hoofdmenu"""
    print("=" * 70)
    print("     KFL BACKEND GPU v5.3 - Backfill Pipeline Menu")
    print("=" * 70)
    print("\nSelecteer een optie:")
    print("  1. 🔄 VOLLEDIGE Backfill (Indicators→Signals→MTF)")
    print("  2. 📝 Edit config.yaml")
    print("  3. 🔍 Test database connectie")
    print("  4. 📦 Archiveer oude logs")
    print("  5. 🔄 MTF Backfill (Multi-TimeFrame)")
    print("  6. ⚠️  ATR Backfill (DEPRECATED - gebruik QBN_v2)")
    print("  7. ❌ Exit")
    return input("\nKeuze [1-7]: ").strip()


def main():
    setup_kfl_logging("docker_menu", project_root=PROJECT_ROOT)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        clear_screen()
        choice = show_menu()
        if choice == '1':
            run_full_backfill_sequence()
        elif choice == '2':
            print("Edit config.yaml handmatig of voeg hier menu-logica toe.")
            input("\nDruk op Enter om terug te gaan...")
        elif choice == '3':
            config = load_config()
            test_database_connection(config)
            input("\nDruk op Enter om terug te gaan...")
        elif choice == '4':
            archive_old_logs()
            input("\nDruk op Enter om terug te gaan...")
        elif choice == '5':
            run_mtf_backfill()
        elif choice == '6':
            run_atr_backfill()
        elif choice == '7':
            sys.exit(0)

if __name__ == "__main__":
    main()
