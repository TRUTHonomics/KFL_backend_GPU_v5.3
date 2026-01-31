"""
Configuration constants for the backfill pipeline.

LOOKBACK_BUFFER: Extra klines to load before start_date for indicator stability.
INTERVAL_MINUTES: Mapping from interval codes to minutes for time_close calculation.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

# =============================================================================
# LOOKBACK CONFIGURATION
# =============================================================================

# REASON: Verhoogd naar 1600 voor gap-forward mode
# Zorgt voor volledige convergentie van EMA200 en DEMA200 (~8x period voor 99% stabiliteit)
# Bij kleinere gaps voorkomt dit indicator afwijkingen aan het begin van de gap
LOOKBACK_BUFFER = 1600

# Minimum klines nodig per indicator voor stabiele output
INDICATOR_WARMUP = {
    'sma_200': 200,
    'ema_200': 200,  # Effectief ~3*200 voor 99% convergentie
    'dema_200': 200,  # Effectief ~400 voor stabiliteit
    'rsi_21': 21,
    'macd_20_50_15': 50 + 15,  # slow + signal
    'atr_14': 14,
    'adx_14': 28,  # 14 + 14 voor ADX smoothing
    'ichimoku_10_30_60': 60,
    'stochastic': 14 + 3,  # k_period + d_smooth
    'bollinger': 20,
    'cmf': 20,
    'supertrend': 14,  # ATR period
    'ao': 34,  # slow period
}

# Maximum warmup needed
MAX_WARMUP = max(INDICATOR_WARMUP.values())

# =============================================================================
# INTERVAL CONFIGURATION
# =============================================================================

# Interval code naar minuten mapping
INTERVAL_MINUTES: Dict[str, int] = {
    '1': 1,
    '3': 3,
    '5': 5,
    '15': 15,
    '30': 30,
    '60': 60,
    '120': 120,
    '240': 240,
    '360': 360,
    '720': 720,
    'D': 1440,
    '1440': 1440,
    'W': 10080,
    '10080': 10080,
    'M': 43200,  # ~30 dagen
    '22222': 43200,
}

# Standaard intervals voor backfill
DEFAULT_INTERVALS = ['D', '240', '60', '1']

import os

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# REASON: Haal database credentials ALTIJD uit environment variabelen (.env.local via Docker)
# Dit is de Single Source of Truth voor connectiegegevens.
DB_USER = os.getenv("POSTGRES_USER") or os.getenv("DB_USER") or "postgres"
DB_PASS = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASS") or "1234"
DB_HOST = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST") or "10.10.10.3"
DB_PORT = os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT") or "5432"
DB_NAME = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME") or "kflhyper"

DEFAULT_DB_CONNECTION = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Batch sizes
DEFAULT_BATCH_SIZE = 500_000  # Rijen per batch voor processing
COPY_CHUNK_SIZE = 100_000  # Rijen per COPY operatie

# =============================================================================
# MTF BACKFILL CONFIGURATIE
# =============================================================================

# REASON: MTF backfill is memory-intensief doordat alle timeframes in RAM worden
# samengevoegd. Conservatieve defaults voorkomen OOM op database server.

# Maximum aantal parallelle workers voor MTF backfill
# REASON: Elke worker houdt meerdere DataFrames in geheugen. 
# Verlaagd naar 4 om OOM te voorkomen bij parallelle joins van grote periodes.
MTF_MAX_WORKERS = 4

# Chunk grootte in dagen voor MTF verwerking
# REASON: 30 dagen 1m data = ~43K rijen per asset, behapbaar voor RAM
MTF_CHUNK_DAYS = 30

# Fetch batch size voor server-side cursors (0 = alles in geheugen)
# REASON: Voorkomt dat PostgreSQL hele resultset in shared_buffers laadt
# REASON: Verlaagd naar 100.000 voor stabielere RAM-druk op DB server
MTF_FETCH_SIZE = 100_000

# =============================================================================
# ATR BACKFILL CONFIGURATIE
# =============================================================================

# Maximum aantal parallelle workers voor ATR backfill
# REASON: Verlaagd van 8 naar 4 om OOM op database server te voorkomen bij joins
# van zeer grote tabellen (100GB+).
ATR_MAX_WORKERS = 4

# Batch size voor ATR backfill (records per batch)
# REASON: Time-based batching wordt gebruikt, maar deze waarde kan gebruikt worden
# voor validatie of toekomstige size-based batching
ATR_BATCH_SIZE = 100_000

# =============================================================================
# INDICATOR KOLOMMEN
# =============================================================================

@dataclass
class IndicatorColumns:
    """Definities van alle indicator kolommen per categorie."""
    
    # RSI kolommen (3)
    RSI: Tuple[str, ...] = ('rsi_7', 'rsi_14', 'rsi_21')
    
    # MACD varianten (5 x 5 = 25)
    MACD_VARIANTS: Tuple[str, ...] = (
        '12_26_9', '6_13_4', '20_50_15', '8_24_9', '5_35_5'
    )
    MACD_SUFFIXES: Tuple[str, ...] = ('', '_signal', '_histogram', '_fast', '_slow')
    
    # Moving averages
    SMA: Tuple[str, ...] = ('sma_20', 'sma_50', 'sma_100', 'sma_200')
    EMA: Tuple[str, ...] = ('ema_10', 'ema_12', 'ema_20', 'ema_26', 'ema_50', 'ema_100', 'ema_200')
    DEMA: Tuple[str, ...] = ('dema_10', 'dema_20', 'dema_50', 'dema_100', 'dema_200')
    
    # Ichimoku varianten (4 x 5 = 20)
    ICHIMOKU_VARIANTS: Tuple[str, ...] = ('9_26_52', '7_22_44', '6_20_52', '10_30_60')
    ICHIMOKU_COMPONENTS: Tuple[str, ...] = ('tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou')
    
    # ATR/ADX (6)
    ATR_ADX: Tuple[str, ...] = ('atr_14', 'atr_10', 'atr_ma', 'adx_14', 'dm_plus_14', 'dm_minus_14')
    
    # Stochastic (2)
    STOCHASTIC: Tuple[str, ...] = ('stoch_k', 'stoch_d')
    
    # Bollinger Bands (3)
    BOLLINGER: Tuple[str, ...] = ('bb_upper', 'bb_middle', 'bb_lower')
    
    # Keltner Channels (3)
    KELTNER: Tuple[str, ...] = ('keltner_upper', 'keltner_middle', 'keltner_lower')
    
    # Volume indicators (4)
    VOLUME: Tuple[str, ...] = ('cmf', 'obv', 'volume_flow', 'vwap_typical_price', 'vwap_close')
    
    # Other (3)
    OTHER: Tuple[str, ...] = ('ao_5_34', 'supertrend_10_3', 'supertrend_direction')
    
    # VPVR (7)
    VPVR: Tuple[str, ...] = (
        'vpvr_poc', 'vpvr_vah', 'vpvr_val',
        'vpvr_hvn_upper', 'vpvr_hvn_lower',
        'vpvr_lvn_upper', 'vpvr_lvn_lower'
    )
    
    @classmethod
    def get_all_macd_columns(cls) -> List[str]:
        """Genereer alle MACD kolom namen."""
        cols = []
        for variant in cls.MACD_VARIANTS:
            for suffix in cls.MACD_SUFFIXES:
                cols.append(f'macd_{variant}{suffix}')
        return cols
    
    @classmethod
    def get_all_ichimoku_columns(cls) -> List[str]:
        """Genereer alle Ichimoku kolom namen."""
        cols = []
        for variant in cls.ICHIMOKU_VARIANTS:
            for component in cls.ICHIMOKU_COMPONENTS:
                cols.append(f'{component}_{variant}')
        return cols
    
    @classmethod
    def get_all_columns(cls) -> List[str]:
        """Alle indicator kolommen in database volgorde."""
        cols = []
        cols.extend(cls.RSI)
        cols.extend(cls.get_all_macd_columns())
        cols.extend(cls.SMA)
        cols.extend(cls.EMA)
        cols.extend(cls.DEMA)
        cols.extend(cls.get_all_ichimoku_columns())
        cols.extend(cls.ATR_ADX)
        cols.extend(cls.STOCHASTIC)
        cols.extend(cls.BOLLINGER)
        cols.extend(cls.KELTNER)
        cols.extend(cls.VOLUME)
        cols.extend(cls.OTHER)
        cols.extend(cls.VPVR)
        return cols


# =============================================================================
# SIGNAL KOLOMMEN
# =============================================================================

@dataclass
class SignalColumns:
    """Definities van alle signal kolommen per classificatie."""
    
    # LEADING signals (48)
    LEAD: Tuple[str, ...] = (
        'kc_mean_reversion_long', 'kc_mean_reversion_short', 'kc_squeeze',
        'kc_dynamic_support', 'kc_dynamic_resistance', 'kc_pullback_long', 'kc_pullback_short',
        'bb_squeeze', 'bb_mean_reversion_long',
        'rsi_oversold', 'rsi_overbought', 'rsi_extreme_oversold', 'rsi_extreme_overbought',
        'rsi_divergence_bearish', 'rsi_divergence_bullish',
        'macd_6_13_4_bullish_cross', 'macd_6_13_4_bearish_cross',
        'macd_8_24_9_bullish_cross', 'macd_8_24_9_bearish_cross',
        'macd_divergence_bullish', 'macd_divergence_bearish',
        'obv_bullish_divergence', 'obv_bearish_divergence',
        'stoch_oversold', 'stoch_overbought',
        'stoch_divergence_bull', 'stoch_divergence_bear',
        'stoch_hidden_divergence_bull', 'stoch_hidden_divergence_bear',
        'ao_twin_peaks_bearish', 'ao_twin_peaks_bullish', 'ao_saucer_bullish', 'ao_saucer_bearish',
        'adx_trend_exhaustion', 'adx_peak_reversal',
        'ichimoku_kumo_twist_bull', 'ichimoku_kumo_twist_bear',
        'ichi_6_20_52_tk_cross_bull', 'ichi_6_20_52_tk_cross_bear',
        'ichi_6_20_52_kumo_breakout_long', 'ichi_6_20_52_kumo_breakout_short',
        'vpvr_hvn_resistance', 'vpvr_hvn_support', 'vpvr_poc_touch',
        'cmf_divergence_bullish', 'cmf_divergence_bearish',
        'mean_reversion_setup_long', 'mean_reversion_setup_short',
    )
    
    # Discrete signals in LEAD table
    LEAD_DISCRETE: Tuple[str, ...] = ('rsi_signal', 'stoch_signal')
    
    # COINCIDENT signals (39)
    COIN: Tuple[str, ...] = (
        'cmf_bullish_bias', 'cmf_bearish_bias', 'cmf_strong_buying', 'cmf_strong_selling',
        'bb_breakout_long', 'bb_breakout_short',
        'macd_bullish_cross', 'macd_bearish_cross', 'macd_histogram_positive', 'macd_histogram_negative',
        'atr_high_volatility', 'atr_low_volatility',
        'stoch_bullish_cross',
        'ao_bullish_zero_cross', 'ao_bearish_zero_cross',
        'volatility_breakout_long', 'volatility_breakout_short',
        'momentum_bullish_confluence', 'momentum_bearish_confluence',
        'volume_bullish_confluence', 'volume_bearish_confluence',
        'risk_high_volatility', 'risk_low_liquidity', 'regime_volatile',
        'di_bullish_cross', 'di_bearish_cross',
        'super_trend_flip_bull', 'super_trend_flip_bear',
        'ichimoku_tenkan_kijun_cross_bull', 'ichimoku_tenkan_kijun_cross_bear',
        'ichimoku_price_above_kumo', 'ichimoku_price_below_kumo', 'ichimoku_price_in_kumo',
        'ichi_7_22_44_tk_cross_bull', 'ichi_7_22_44_tk_cross_bear',
        'vpvr_lvn_breakout_up', 'vpvr_lvn_breakout_down',
        'rsi_center_cross_bull', 'rsi_center_cross_bear',
    )
    
    # Discrete signals in COIN table
    COIN_DISCRETE: Tuple[str, ...] = ('macd_signal', 'cmf_signal', 'bb_signal', 'keltner_signal', 'atr_signal')
    
    # CONFIRMING signals (38)
    CONF: Tuple[str, ...] = (
        'adx_trend_confirm', 'adx_non_trending_regime', 'adx_strong_trend', 'adx_weak_trend',
        'kc_trend_breakout_long', 'kc_trend_breakout_short',
        'macd_5_35_5_bullish_cross', 'macd_5_35_5_bearish_cross',
        'macd_20_50_15_bullish_cross', 'macd_20_50_15_bearish_cross',
        'macd_zero_line_cross_bull', 'macd_zero_line_cross_bear',
        'bullish_confluence_strong', 'bearish_confluence_strong',
        'trend_following_breakout_long', 'trend_following_breakout_short',
        'mtf_bullish_alignment', 'mtf_bearish_alignment',
        'regime_trending_bullish', 'regime_trending_bearish', 'regime_ranging',
        'di_strong_bullish', 'di_strong_bearish',
        'super_trend_bullish', 'super_trend_bearish',
        'ichimoku_kumo_breakout_long', 'ichimoku_kumo_breakout_short',
        'ichi_10_30_60_tk_cross_bull', 'ichi_10_30_60_tk_cross_bear',
        'ichi_10_30_60_kumo_breakout_long', 'ichi_10_30_60_kumo_breakout_short',
        'vpvr_value_area_inside',
        'obv_trend_confirm_bull', 'obv_trend_confirm_bear',
        'obv_trend_strength_bull', 'obv_trend_strength_bear',
        'rsi_center_bullish', 'rsi_center_bearish',
    )
    
    # Discrete signals in CONF table
    CONF_DISCRETE: Tuple[str, ...] = ('adx_signal',)
    
    # Concordance kolommen (in alle tabellen)
    CONCORDANCE: Tuple[str, ...] = ('concordance_sum', 'concordance_count', 'concordance_score')


# Singleton instances
INDICATOR_COLS = IndicatorColumns()
SIGNAL_COLS = SignalColumns()
