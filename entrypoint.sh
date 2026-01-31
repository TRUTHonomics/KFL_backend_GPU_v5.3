#!/bin/bash
set -e

echo "=========================================="
echo "GPU Container v5 Entrypoint"
echo "=========================================="

echo "Initialiseer database parameters..."
python3 /app/init_db_params.py

echo "Database parameters geinitialiseerd"
echo "Start container..."
exec "$@"
