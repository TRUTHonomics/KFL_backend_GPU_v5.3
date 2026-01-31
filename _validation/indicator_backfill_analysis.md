# Rapport: Analyse Lege Indicator Kolommen (GPU Backfill)

Dit rapport beschrijft de oorzaken van de structureel lege kolommen in de `kfl.indicators` tabel na een run van de `KFL_backend_GPU_v5.3` pipeline (`source_script = 'GPU_backfill'`).

## 1. Samenvatting Probleemkolommen
Op basis van een SQL analyse op ~3.1 miljoen rijen zijn de volgende kolommen structureel leeg (0 rijen gevuld):

| Categorie | Lege Kolommen |
| :--- | :--- |
| **ADX/DI** | `adx_14` (CRITICAL) |
| **MACD** | `macd_20_50_15`, `macd_8_24_9`, `macd_5_35_5` (incl. componenten) |
| **RSI** | `rsi_21` |
| **EMA/DEMA** | `ema_12`, `ema_20`, `dema_10`, `dema_20`, `dema_50`, `dema_100`, `dema_200` |
| **Volume** | `volume_flow`, `vwap_typical_price`, `vwap_close` |

---

## 2. Diagnoses en Oorzaken

### A. De ADX "NaN" Bug (Kritiek)
**Oorzaak:** De Numba kernels `_ema_core` en `_rma_core` in `src/backfill/indicators/cpu_indicators.py` kunnen niet omgaan met leading `NaN` waarden.
- ADX berekent eerst de `dx` array. Deze array heeft altijd NaNs aan het begin vanwege de warmup van ATR en DI.
- De kernels initialiseren met een SMA over de eerste `period` rijen. Als daar een `NaN` tussen zit, wordt de gehele resultaat-array `NaN`.
- Dit verklaart waarom `dm_plus_14` wel gevuld is, maar `adx_14` leeg blijft.

### B. Missing Writer Mappings
**Oorzaak:** De `IndicatorWriter` in `src/backfill/writers/indicator_writer.py` is niet synchroon met het database schema.
- De lijst `INDICATOR_COLUMNS` bevat niet alle MACD varianten die wel in de tabel staan.
- Zonder vermelding in deze lijst worden kolommen simpelweg overgeslagen tijdens de `COPY` operatie.

### C. Incomplete Calculator Logica
**Oorzaak:** Sommige indicatoren worden simpelweg niet berekend in `cpu_indicators.py` of `gpu_indicators.py`.
- `rsi_21`: De calculator doet alleen 7 en 14.
- `volume_flow`, `vwap_*`: Geen implementatie aanwezig.

### D. EMA/DEMA Periode Mismatch
**Oorzaak:** De calculator berekent een hardcoded set periodes `[7, 25, 50, 99, 200]`, terwijl de database/writer andere periodes verwacht (`ema_10`, `ema_12`, etc.).

---

## 3. Herstelvoorstellen (Concrete Fixes)

### Fix 1: Robuuste Numba Kernels
Pas `_ema_core` en `_rma_core` aan in `src/backfill/indicators/cpu_indicators.py` om de eerste geldige waarde te zoeken:

```python
@jit(nopython=True, cache=True)
def _ema_core(data: np.ndarray, period: int) -> np.ndarray:
    # Zoek eerste non-NaN index
    start_idx = -1
    for i in range(len(data)):
        if not np.isnan(data[i]):
            start_idx = i
            break
    
    if start_idx == -1 or len(data) < start_idx + period:
        return result # result met NaNs
    
    # Start berekening vanaf start_idx
    # ... SMA initialisatie vanaf start_idx ...
```

### Fix 2: Update IndicatorWriter
Voeg alle ontbrekende kolommen toe aan `INDICATOR_COLUMNS` in `src/backfill/writers/indicator_writer.py`.

### Fix 3: Synchroniseer Periodes
Update `calculate_all_emas` en `calculate_all_demas` in `src/backfill/indicators/cpu_indicators.py` om de periodes te berekenen die de database verwacht.

### Fix 4: Implementeer RSI 21
Voeg periode `21` toe aan de `periods` array in `calculate_rsi_variants`.

---

## 4. Impact op QBN
Het ontbreken van `adx_14` zorgt ervoor dat de `HTF_Regime` node in het Bayesian Network blind is. Dit resulteert in een entropy van 0.0 en nul voorspellende waarde. Het herstellen van ADX is de hoogste prioriteit.

