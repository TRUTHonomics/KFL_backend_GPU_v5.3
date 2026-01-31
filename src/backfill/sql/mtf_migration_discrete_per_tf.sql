-- =============================================================================
-- MTF Migration: Discrete Signals en Concordance per Timeframe
-- =============================================================================
-- 
-- DOEL: Alle kolommen met timeframe suffix voor consistentie
-- 
-- Oude structuur (inconsistent):
--   - Boolean signals: rsi_oversold_d, rsi_oversold_240, rsi_oversold_60, rsi_oversold_1
--   - Discrete signals: rsi_signal (alleen 1m, geen suffix)
--   - Concordance: concordance_sum (alleen 60m, geen suffix)
--
-- Nieuwe structuur (consistent):
--   - Boolean signals: rsi_oversold_d, rsi_oversold_240, rsi_oversold_60, rsi_oversold_1
--   - Discrete signals: rsi_signal_d, rsi_signal_240, rsi_signal_60, rsi_signal_1
--   - Concordance: concordance_sum_d, concordance_sum_240, concordance_sum_60, concordance_sum_1
--
-- TIMEFRAMES: d=Daily, 240=4H, 60=1H, 1=1m
-- =============================================================================

-- =============================================================================
-- DEEL 1: NIEUWE KOLOMMEN TOEVOEGEN
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1.1 mtf_signals_lead
-- -----------------------------------------------------------------------------
-- Discrete: rsi_signal, stoch_signal (2 x 4 TF = 8 kolommen)
-- Concordance: sum, count, score (3 x 4 TF = 12 kolommen)
-- Totaal: 20 nieuwe kolommen

ALTER TABLE kfl.mtf_signals_lead
    -- Discrete signals per timeframe
    ADD COLUMN IF NOT EXISTS rsi_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS rsi_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS rsi_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS rsi_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS stoch_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS stoch_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS stoch_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS stoch_signal_1 SMALLINT,
    -- Concordance per timeframe
    ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;

-- -----------------------------------------------------------------------------
-- 1.2 mtf_signals_coin
-- -----------------------------------------------------------------------------
-- Discrete: macd_signal, cmf_signal, bb_signal, keltner_signal, atr_signal (5 x 4 TF = 20 kolommen)
-- Concordance: 3 x 4 TF = 12 kolommen
-- Totaal: 32 nieuwe kolommen

ALTER TABLE kfl.mtf_signals_coin
    -- Discrete signals per timeframe
    ADD COLUMN IF NOT EXISTS macd_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS macd_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS macd_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS macd_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS cmf_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS cmf_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS cmf_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS cmf_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS bb_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS bb_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS bb_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS bb_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS keltner_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS keltner_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS keltner_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS keltner_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS atr_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS atr_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS atr_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS atr_signal_1 SMALLINT,
    -- Concordance per timeframe
    ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;

-- -----------------------------------------------------------------------------
-- 1.3 mtf_signals_conf
-- -----------------------------------------------------------------------------
-- Discrete: adx_signal (1 x 4 TF = 4 kolommen)
-- Concordance: 3 x 4 TF = 12 kolommen
-- Totaal: 16 nieuwe kolommen

ALTER TABLE kfl.mtf_signals_conf
    -- Discrete signals per timeframe
    ADD COLUMN IF NOT EXISTS adx_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS adx_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS adx_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS adx_signal_1 SMALLINT,
    -- Concordance per timeframe
    ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;

-- =============================================================================
-- DEEL 1B: CURRENT TABELLEN UITBREIDEN (real-time triggers schrijven hierheen)
-- =============================================================================

-- mtf_signals_current_lead
ALTER TABLE kfl.mtf_signals_current_lead
    ADD COLUMN IF NOT EXISTS rsi_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS rsi_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS rsi_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS rsi_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS stoch_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS stoch_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS stoch_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS stoch_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;

-- mtf_signals_current_coin
ALTER TABLE kfl.mtf_signals_current_coin
    ADD COLUMN IF NOT EXISTS macd_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS macd_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS macd_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS macd_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS cmf_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS cmf_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS cmf_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS cmf_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS bb_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS bb_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS bb_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS bb_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS keltner_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS keltner_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS keltner_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS keltner_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS atr_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS atr_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS atr_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS atr_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;

-- mtf_signals_current_conf
ALTER TABLE kfl.mtf_signals_current_conf
    ADD COLUMN IF NOT EXISTS adx_signal_d SMALLINT,
    ADD COLUMN IF NOT EXISTS adx_signal_240 SMALLINT,
    ADD COLUMN IF NOT EXISTS adx_signal_60 SMALLINT,
    ADD COLUMN IF NOT EXISTS adx_signal_1 SMALLINT,
    ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
    ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;

-- =============================================================================
-- DEEL 2: ARCHIVE TABELLEN UITBREIDEN (indien aanwezig)
-- =============================================================================

-- Archive tabellen hebben dezelfde structuur als de MTF tabellen
-- Deze moeten ook worden uitgebreid met de nieuwe kolommen

DO $$
BEGIN
    -- Check if mtf_signals_lead_archive exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kfl' AND table_name = 'mtf_signals_lead_archive') THEN
        ALTER TABLE kfl.mtf_signals_lead_archive
            ADD COLUMN IF NOT EXISTS rsi_signal_d SMALLINT,
            ADD COLUMN IF NOT EXISTS rsi_signal_240 SMALLINT,
            ADD COLUMN IF NOT EXISTS rsi_signal_60 SMALLINT,
            ADD COLUMN IF NOT EXISTS rsi_signal_1 SMALLINT,
            ADD COLUMN IF NOT EXISTS stoch_signal_d SMALLINT,
            ADD COLUMN IF NOT EXISTS stoch_signal_240 SMALLINT,
            ADD COLUMN IF NOT EXISTS stoch_signal_60 SMALLINT,
            ADD COLUMN IF NOT EXISTS stoch_signal_1 SMALLINT,
            ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;
    END IF;
    
    -- Check if mtf_signals_coin_archive exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kfl' AND table_name = 'mtf_signals_coin_archive') THEN
        ALTER TABLE kfl.mtf_signals_coin_archive
            ADD COLUMN IF NOT EXISTS macd_signal_d SMALLINT,
            ADD COLUMN IF NOT EXISTS macd_signal_240 SMALLINT,
            ADD COLUMN IF NOT EXISTS macd_signal_60 SMALLINT,
            ADD COLUMN IF NOT EXISTS macd_signal_1 SMALLINT,
            ADD COLUMN IF NOT EXISTS cmf_signal_d SMALLINT,
            ADD COLUMN IF NOT EXISTS cmf_signal_240 SMALLINT,
            ADD COLUMN IF NOT EXISTS cmf_signal_60 SMALLINT,
            ADD COLUMN IF NOT EXISTS cmf_signal_1 SMALLINT,
            ADD COLUMN IF NOT EXISTS bb_signal_d SMALLINT,
            ADD COLUMN IF NOT EXISTS bb_signal_240 SMALLINT,
            ADD COLUMN IF NOT EXISTS bb_signal_60 SMALLINT,
            ADD COLUMN IF NOT EXISTS bb_signal_1 SMALLINT,
            ADD COLUMN IF NOT EXISTS keltner_signal_d SMALLINT,
            ADD COLUMN IF NOT EXISTS keltner_signal_240 SMALLINT,
            ADD COLUMN IF NOT EXISTS keltner_signal_60 SMALLINT,
            ADD COLUMN IF NOT EXISTS keltner_signal_1 SMALLINT,
            ADD COLUMN IF NOT EXISTS atr_signal_d SMALLINT,
            ADD COLUMN IF NOT EXISTS atr_signal_240 SMALLINT,
            ADD COLUMN IF NOT EXISTS atr_signal_60 SMALLINT,
            ADD COLUMN IF NOT EXISTS atr_signal_1 SMALLINT,
            ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;
    END IF;
    
    -- Check if mtf_signals_conf_archive exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'kfl' AND table_name = 'mtf_signals_conf_archive') THEN
        ALTER TABLE kfl.mtf_signals_conf_archive
            ADD COLUMN IF NOT EXISTS adx_signal_d SMALLINT,
            ADD COLUMN IF NOT EXISTS adx_signal_240 SMALLINT,
            ADD COLUMN IF NOT EXISTS adx_signal_60 SMALLINT,
            ADD COLUMN IF NOT EXISTS adx_signal_1 SMALLINT,
            ADD COLUMN IF NOT EXISTS concordance_sum_d INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_240 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_60 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_sum_1 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_d INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_240 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_60 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_count_1 INTEGER,
            ADD COLUMN IF NOT EXISTS concordance_score_d DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_240 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_60 DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS concordance_score_1 DOUBLE PRECISION;
    END IF;
END $$;

-- =============================================================================
-- DEEL 3: VERIFICATIE
-- =============================================================================

-- Controleer dat alle nieuwe kolommen bestaan
SELECT 
    'mtf_signals_lead' as table_name,
    COUNT(*) FILTER (WHERE column_name LIKE '%_signal_%' OR column_name LIKE 'concordance_%') as new_columns
FROM information_schema.columns 
WHERE table_schema = 'kfl' AND table_name = 'mtf_signals_lead'
UNION ALL
SELECT 
    'mtf_signals_coin',
    COUNT(*) FILTER (WHERE column_name LIKE '%_signal_%' OR column_name LIKE 'concordance_%')
FROM information_schema.columns 
WHERE table_schema = 'kfl' AND table_name = 'mtf_signals_coin'
UNION ALL
SELECT 
    'mtf_signals_conf',
    COUNT(*) FILTER (WHERE column_name LIKE '%_signal_%' OR column_name LIKE 'concordance_%')
FROM information_schema.columns 
WHERE table_schema = 'kfl' AND table_name = 'mtf_signals_conf';

-- =============================================================================
-- DEEL 4: OUDE KOLOMMEN VERWIJDEREN (UITVOEREN NA BACKFILL EN TRIGGER UPDATE)
-- =============================================================================
-- 
-- WAARSCHUWING: Voer dit pas uit NADAT:
-- 1. De triggers zijn bijgewerkt naar de nieuwe kolomstructuur
-- 2. De backfill is uitgevoerd
-- 3. De data is geverifieerd
--
-- Uncomment onderstaande statements om oude kolommen te verwijderen:
--
-- ALTER TABLE kfl.mtf_signals_lead
--     DROP COLUMN IF EXISTS rsi_signal,
--     DROP COLUMN IF EXISTS stoch_signal,
--     DROP COLUMN IF EXISTS concordance_sum,
--     DROP COLUMN IF EXISTS concordance_count,
--     DROP COLUMN IF EXISTS concordance_score;
--
-- ALTER TABLE kfl.mtf_signals_coin
--     DROP COLUMN IF EXISTS macd_signal,
--     DROP COLUMN IF EXISTS cmf_signal,
--     DROP COLUMN IF EXISTS bb_signal,
--     DROP COLUMN IF EXISTS keltner_signal,
--     DROP COLUMN IF EXISTS atr_signal,
--     DROP COLUMN IF EXISTS concordance_sum,
--     DROP COLUMN IF EXISTS concordance_count,
--     DROP COLUMN IF EXISTS concordance_score;
--
-- ALTER TABLE kfl.mtf_signals_conf
--     DROP COLUMN IF EXISTS adx_signal,
--     DROP COLUMN IF EXISTS concordance_sum,
--     DROP COLUMN IF EXISTS concordance_count,
--     DROP COLUMN IF EXISTS concordance_score;
--
-- Archive tabellen (indien aanwezig):
-- ALTER TABLE kfl.mtf_signals_lead_archive ...
-- ALTER TABLE kfl.mtf_signals_coin_archive ...
-- ALTER TABLE kfl.mtf_signals_conf_archive ...
