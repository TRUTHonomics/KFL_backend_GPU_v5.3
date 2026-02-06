#!/bin/bash
# REASON: Set -e verwijderd zodat DB init failure niet de hele container crasht
echo "=========================================="
echo "GPU Container v6 Entrypoint"
echo "=========================================="

echo "Initialiseer database parameters..."
python3 /app/init_db_params.py || echo "⚠️  Database init gefaald - container start toch (DB is mogelijk niet beschikbaar)"

echo "Container gestart - gebruik docker exec voor interactieve sessies"
exec "$@"
