-- ============================================================================
-- MTF Backfill Verification: Lookahead Bias Check
-- 
-- Een MTF rij mag GEEN signalen bevatten van candles die op dat moment
-- nog niet gesloten waren.
--
-- REGEL: Voor elke MTF rij met time_close_1 = T:
--   - time_close_d <= T   (daily candle was al gesloten)
--   - time_close_240 <= T (4H candle was al gesloten)
--   - time_close_60 <= T  (1H candle was al gesloten)
--
-- VERWACHT RESULTAAT: 0 violations in alle queries
-- ============================================================================

-- ============================================================================
-- 1. LEAD tabel verificatie
-- ============================================================================
SELECT 
    'kfl.mtf_signals_lead' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) FILTER (WHERE time_close_d > time_close_1) as d_violations,
    COUNT(*) FILTER (WHERE time_close_240 > time_close_1) as h4_violations,
    COUNT(*) FILTER (WHERE time_close_60 > time_close_1) as h1_violations,
    COUNT(*) FILTER (
        WHERE time_close_d > time_close_1 
           OR time_close_240 > time_close_1 
           OR time_close_60 > time_close_1
    ) as total_violations
FROM kfl.mtf_signals_lead;

-- ============================================================================
-- 2. COIN tabel verificatie
-- ============================================================================
SELECT 
    'kfl.mtf_signals_coin' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) FILTER (WHERE time_close_d > time_close_1) as d_violations,
    COUNT(*) FILTER (WHERE time_close_240 > time_close_1) as h4_violations,
    COUNT(*) FILTER (WHERE time_close_60 > time_close_1) as h1_violations,
    COUNT(*) FILTER (
        WHERE time_close_d > time_close_1 
           OR time_close_240 > time_close_1 
           OR time_close_60 > time_close_1
    ) as total_violations
FROM kfl.mtf_signals_coin;

-- ============================================================================
-- 3. CONF tabel verificatie
-- ============================================================================
SELECT 
    'kfl.mtf_signals_conf' as table_name,
    COUNT(*) as total_rows,
    COUNT(*) FILTER (WHERE time_close_d > time_close_1) as d_violations,
    COUNT(*) FILTER (WHERE time_close_240 > time_close_1) as h4_violations,
    COUNT(*) FILTER (WHERE time_close_60 > time_close_1) as h1_violations,
    COUNT(*) FILTER (
        WHERE time_close_d > time_close_1 
           OR time_close_240 > time_close_1 
           OR time_close_60 > time_close_1
    ) as total_violations
FROM kfl.mtf_signals_conf;

-- ============================================================================
-- 4. Sample van violations (indien aanwezig)
-- ============================================================================
SELECT 
    asset_id,
    time_1,
    time_close_1,
    time_d,
    time_close_d,
    time_240,
    time_close_240,
    time_60,
    time_close_60,
    CASE 
        WHEN time_close_d > time_close_1 THEN 'D lookahead'
        WHEN time_close_240 > time_close_1 THEN '240 lookahead'
        WHEN time_close_60 > time_close_1 THEN '60 lookahead'
    END as violation_type
FROM kfl.mtf_signals_lead
WHERE time_close_d > time_close_1 
   OR time_close_240 > time_close_1 
   OR time_close_60 > time_close_1
LIMIT 10;

-- ============================================================================
-- 5. Voorbeeld: Correcte MTF rij (geen lookahead)
-- ============================================================================
-- Voor een 1m candle om 14:05 (time_close = 14:06):
-- - Meest recente D candle: gisteren (time_close = vandaag 00:00) ✓
-- - Meest recente 4H candle: 12:00-16:00 zou pas om 16:00 sluiten ✗
--   → Gebruik 08:00-12:00 candle (time_close = 12:00) ✓
-- - Meest recente 1H candle: 14:00-15:00 sluit pas om 15:00 ✗
--   → Gebruik 13:00-14:00 candle (time_close = 14:00) ✓

SELECT 
    asset_id,
    time_1,
    time_close_1 as "1m_close (beschikbaar op)",
    time_d || ' - ' || time_close_d as "D candle (time - close)",
    time_240 || ' - ' || time_close_240 as "4H candle (time - close)",
    time_60 || ' - ' || time_close_60 as "1H candle (time - close)"
FROM kfl.mtf_signals_lead
WHERE asset_id = 1
ORDER BY time_1 DESC
LIMIT 5;
