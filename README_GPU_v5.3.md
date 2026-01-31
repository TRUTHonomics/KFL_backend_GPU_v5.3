# KFL Backend GPU v5 - Indicator & Signal Backfill Pipeline

High-performance backfill pipeline voor indicatoren en signals in KlineFuturesLab.

## Features

- **Hybrid GPU/CPU berekeningen**: Vectorized indicators op GPU (CuPy), recursive indicators op CPU (Numba)
- **100M+ rows ondersteuning**: Geoptimaliseerd voor grote datasets
- **DB-Driven Configuration**: Laadt discretisatie drempelwaarden direct uit de database (`qbn.signal_discretization`).
- **Staging table strategie**: COPY FROM STDIN + UPSERT voor maximale write performance
- **Gap detection**: Alleen ontbrekende data bijvullen (gaps_only mode)
- **time_close berekening**: Correct voor lookahead bias preventie

## Architectuur (Configuration)

### DB-Driven Threshold Configuration

Sinds 2026-01-04 is deze container database-driven voor signal definities en discretisatie:

| Configuratie | Database Tabel | Fallback | Update via |
|--------------|----------------|----------|------------|
| Indicator Discretization | `qbn.signal_discretization` | Hardcoded defaults | QBN v3 `db_sync.py` |
| Signal Classification | `qbn.signal_classification` | N/A | QBN v3 `db_sync.py` |

**Data Flow:**
1. **Init Phase**: `DiscreteSignalCalculator` laadt thresholds uit `qbn.signal_discretization`
2. **Consistency**: Zelfde discrete waarden als real-time pipeline (KFL v3)
3. **Source of Truth**: Database (`qbn` schema) is leidend

**BELANGRIJK**: De composite thresholds (`COMPOSITE_NEUTRAL_BAND`, `COMPOSITE_STRONG_THRESHOLD`) zijn niet relevant voor deze container. Die worden alleen gebruikt in QBN v3 voor Bayesian network inference, niet voor indicator discretisatie.

## Signals

### Discrete Signals (8) - Waarden -2 tot +2
De drempelwaarden voor deze signalen zijn centraal aanpasbaar in `discretization.yaml` (v3):
1. `rsi_discrete`
2. `macd_discrete`
3. `bb_discrete`
4. `stoch_discrete`
5. `adx_trend_discrete`
6. `supertrend_discrete`
7. `ichimoku_discrete`
8. `ao_discrete`

### Boolean Signals (125) - Waarden -1, 0, +1
Definities worden geladen uit `qbn.signal_classification` (gesynchroniseerd vanuit `signals.yaml`).

## Database Schema

### qbn.signal_discretization
Centrale tabel voor alle indicator drempelwaarden. 
Kolommen: `indicator_base`, `threshold_name`, `threshold_value`.

### qbn.signal_classification
Centrale tabel voor alle signal metadata en semantische klassen.

## Gebruik

### Command Line
```bash
# Full backfill (gebruikt DB thresholds)
python -m backfill.cli --asset_id 1 --interval_min 1 --mode full
```

### Python API
```python
from backfill.pipeline import BackfillPipeline

# Initialiseert automatisch DB connectie voor thresholds
pipeline = BackfillPipeline(connection_string="...")
```
