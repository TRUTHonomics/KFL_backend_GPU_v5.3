#!/usr/bin/env python3
"""
Database Parameter Initialization Script
=========================================
Stelt TimescaleDB parameters in bij het opstarten van de GPU container.
Dit voorkomt 'tuple decompression limit exceeded' errors tijdens backfill operaties.
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import sql
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('init_db_params')


def init_timescaledb_params():
    """Stel TimescaleDB parameters in voor alle sessies."""
    db_host = os.getenv('POSTGRES_HOST', 'postgres-server')
    db_port = int(os.getenv('POSTGRES_PORT', '5432'))
    db_name = os.getenv('POSTGRES_DB', 'KFLhyper')
    db_user = os.getenv('POSTGRES_USER', 'pipeline')
    db_password = os.getenv('POSTGRES_PASSWORD', 'pipeline123')
    
    conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"
    
    try:
        logger.info(f"Verbinden met database {db_host}:{db_port}/{db_name}...")
        conn = psycopg2.connect(conn_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check TimescaleDB
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_extension 
                WHERE extname = 'timescaledb'
            )
        """)
        timescale_available = cursor.fetchone()[0]
        
        if not timescale_available:
            logger.warning("TimescaleDB niet beschikbaar - skip parameter initialisatie")
            conn.close()
            return True
        
        logger.info("TimescaleDB gevonden - stel parameters in...")
        
        # Parameters voor grote UPSERT operaties
        params_to_set = [
            ("timescaledb.max_tuples_decompressed_per_dml_transaction", "0", 
             "Unlimited tuple decompression voor grote UPSERT operaties"),
        ]
        
        for param_name, param_value, description in params_to_set:
            try:
                alter_query = sql.SQL("ALTER DATABASE {} SET {} = {}").format(
                    sql.Identifier(db_name),
                    sql.Identifier(param_name),
                    sql.Literal(param_value)
                )
                cursor.execute(alter_query)
                logger.info(f"✅ {param_name} = {param_value} ({description})")
            except psycopg2.errors.InsufficientPrivilege:
                logger.warning(f"⚠️  Geen superuser privileges - stel {param_name} alleen in voor huidige sessie")
                try:
                    set_query = sql.SQL("SET {} = {}").format(
                        sql.Identifier(param_name),
                        sql.Literal(param_value)
                    )
                    cursor.execute(set_query)
                    logger.info(f"✅ {param_name} = {param_value} (alleen huidige sessie)")
                except Exception as e2:
                    logger.error(f"❌ Kon {param_name} niet instellen: {e2}")
            except Exception as e:
                logger.warning(f"⚠️  Kon {param_name} niet instellen via ALTER DATABASE: {e}")
        
        # Verifieer
        cursor.execute("SHOW timescaledb.max_tuples_decompressed_per_dml_transaction")
        current_value = cursor.fetchone()[0]
        logger.info(f"📊 Huidige waarde: {current_value}")
        
        conn.close()
        logger.info("✅ Database parameter initialisatie voltooid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Fout bij database parameter initialisatie: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Database Parameter Initialization Script")
    logger.info("=" * 60)
    
    success = init_timescaledb_params()
    sys.exit(0 if success else 1)
