-- ============================================================================
-- MTF Backfill Indexes
-- Optimaliseert time_bucket() LEFT JOIN queries voor lookahead-vrije MTF backfill
-- 
-- De MTF backfill gebruikt set-based joins met time_bucket() voor O(N+M) performance.
-- Deze indexen versnellen de lookup van hogere timeframe candles per interval.
-- 
-- Gebruik: psql -f mtf_backfill_indexes.sql
-- ============================================================================

-- Index voor signals_lead: bucket join op (asset_id, interval_min, time_close)
-- REASON: Optimaliseert LEFT JOIN met time_bucket() condition
-- Query: ... LEFT JOIN signals_lead tf_d ON tf_d.time_close = time_bucket('1 day', base.time_close)
-- NOTE: Geen CONCURRENTLY - hypertables ondersteunen dit niet
CREATE INDEX IF NOT EXISTS idx_signals_lead_mtf_lookup
ON kfl.signals_lead (asset_id, interval_min, time_close DESC);

-- Index voor signals_coin: zelfde structuur voor bucket joins
CREATE INDEX IF NOT EXISTS idx_signals_coin_mtf_lookup
ON kfl.signals_coin (asset_id, interval_min, time_close DESC);

-- Index voor signals_conf: zelfde structuur voor bucket joins
CREATE INDEX IF NOT EXISTS idx_signals_conf_mtf_lookup
ON kfl.signals_conf (asset_id, interval_min, time_close DESC);

-- ============================================================================
-- Verificatie
-- ============================================================================
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'kfl'
  AND indexname LIKE '%mtf_lookup%'
ORDER BY tablename;
