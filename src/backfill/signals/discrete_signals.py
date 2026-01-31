"""
Discrete signal calculations.

Maps continuous indicator values to discrete signals:
-2 = strong bearish
-1 = weak bearish
 0 = neutral
+1 = weak bullish
+2 = strong bullish

REASON: Thresholds are loaded from qbn.signal_discretization in the database.
"""

import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DiscreteSignalCalculator:
    """
    Calculates discrete signals from indicator values.
    
    Thresholds are loaded from the database at initialization.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize discrete signal calculator.
        
        Args:
            connection_string: Optional DB connection string to load thresholds
        """
        self._thresholds: Dict[str, Dict[str, float]] = {}
        if connection_string:
            self._load_thresholds(connection_string)
        else:
            logger.warning("No connection_string provided to DiscreteSignalCalculator, using hardcoded defaults.")
            
        logger.info("Discrete signal calculator initialized")

    def _load_thresholds(self, connection_string: str):
        """Load drempelwaarden uit qbn.signal_discretization."""
        import psycopg2
        try:
            with psycopg2.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT indicator_base, threshold_name, threshold_value 
                        FROM qbn.signal_discretization
                    """)
                    rows = cur.fetchall()
                    for ind, name, val in rows:
                        if ind not in self._thresholds:
                            self._thresholds[ind] = {}
                        self._thresholds[ind][name] = float(val)
            logger.info(f"✅ {len(rows)} discretisatie thresholds geladen uit database")
        except Exception as e:
            logger.error(f"❌ Fout bij laden thresholds uit database: {e}")

    # =========================================================================
    # LEAD table discrete signals
    # =========================================================================
    
    def rsi_signal(self, rsi: np.ndarray) -> np.ndarray:
        """Map RSI to discrete signal."""
        n = len(rsi)
        result = np.zeros(n, dtype=np.int8)
        
        t = self._thresholds.get('RSI', {})
        ext_os = t.get('extreme_oversold', 20.0)
        os = t.get('oversold', 30.0)
        ext_ob = t.get('extreme_overbought', 80.0)
        ob = t.get('overbought', 70.0)
        
        # REASON: contrarian logic - overbought = bearish, oversold = bullish
        result[rsi < ext_os] = 2
        result[(rsi >= ext_os) & (rsi < os)] = 1
        result[rsi > ext_ob] = -2
        result[(rsi > ob) & (rsi <= ext_ob)] = -1
        
        result[np.isnan(rsi)] = 0
        return result
    
    def stoch_signal(self, stoch_k: np.ndarray) -> np.ndarray:
        """Map Stochastic Oscillator to discrete signal."""
        n = len(stoch_k)
        result = np.zeros(n, dtype=np.int8)
        
        t = self._thresholds.get('STOCHASTIC', {})
        ext_os = t.get('extreme_oversold', 10.0)
        os = t.get('oversold', 20.0)
        ext_ob = t.get('extreme_overbought', 90.0)
        ob = t.get('overbought', 80.0)
        
        result[stoch_k < ext_os] = 2
        result[(stoch_k >= ext_os) & (stoch_k < os)] = 1
        result[stoch_k > ext_ob] = -2
        result[(stoch_k > ob) & (stoch_k <= ext_ob)] = -1
        
        result[np.isnan(stoch_k)] = 0
        return result
    
    # =========================================================================
    # COIN table discrete signals
    # =========================================================================
    
    def macd_signal(
        self,
        macd_line: np.ndarray,
        macd_signal_line: np.ndarray,
        macd_hist: np.ndarray
    ) -> np.ndarray:
        """Map MACD to discrete signal."""
        n = len(macd_line)
        result = np.zeros(n, dtype=np.int8)
        
        t = self._thresholds.get('MACD', {})
        hist_strong = t.get('histogram_strong_threshold', 0.001)
        ratio_strong = t.get('signal_ratio_strong', 1.1)
        
        # Bullish conditions
        bullish = macd_hist > 0
        # Vermeiden van division by zero bij ratio check
        valid_signal = np.abs(macd_signal_line) > 1e-10
        strong_bull = bullish & (macd_hist > hist_strong) & valid_signal & (macd_line > macd_signal_line * ratio_strong)
        
        # Bearish conditions
        bearish = macd_hist < 0
        strong_bear = bearish & (macd_hist < -hist_strong) & valid_signal & (macd_line < macd_signal_line * (2 - ratio_strong))
        
        result[bullish] = 1
        result[strong_bull] = 2
        result[bearish] = -1
        result[strong_bear] = -2
        
        nan_mask = np.isnan(macd_line) | np.isnan(macd_signal_line) | np.isnan(macd_hist)
        result[nan_mask] = 0
        return result
    
    def cmf_signal(self, cmf: np.ndarray) -> np.ndarray:
        """Map Chaikin Money Flow to discrete signal."""
        n = len(cmf)
        result = np.zeros(n, dtype=np.int8)
        
        t = self._thresholds.get('CMF', {})
        strong_buy = t.get('strong_buying', 0.15)
        bull_bias = t.get('bullish_bias', 0.05)
        strong_sell = t.get('strong_selling', -0.15)
        bear_bias = t.get('bearish_bias', -0.05)
        
        result[cmf > strong_buy] = 2
        result[(cmf > bull_bias) & (cmf <= strong_buy)] = 1
        result[(cmf >= bear_bias) & (cmf <= bull_bias)] = 0
        result[(cmf >= strong_sell) & (cmf < bear_bias)] = -1
        result[cmf < strong_sell] = -2
        
        result[np.isnan(cmf)] = 0
        return result
    
    def bb_signal(
        self,
        close: np.ndarray,
        bb_upper: np.ndarray,
        bb_lower: np.ndarray,
        bb_middle: np.ndarray
    ) -> np.ndarray:
        """Map Bollinger Band position to discrete signal."""
        n = len(close)
        result = np.zeros(n, dtype=np.int8)
        
        band_width = bb_upper - bb_lower
        valid = band_width > 0
        position = np.zeros(n)
        position[valid] = (close[valid] - bb_lower[valid]) / band_width[valid]
        
        t = self._thresholds.get('BOLLINGER', {})
        lower_zone = t.get('lower_zone', 0.25)
        upper_zone = t.get('upper_zone', 0.75)
        
        # contrarian logic
        result[position < 0] = 2
        result[(position >= 0) & (position < lower_zone)] = 1
        result[position > 1.0] = -2
        result[(position > upper_zone) & (position <= 1.0)] = -1
        
        nan_mask = np.isnan(bb_upper) | np.isnan(bb_lower) | np.isnan(bb_middle) | ~valid
        result[nan_mask] = 0
        return result
    
    def keltner_signal(
        self,
        close: np.ndarray,
        kc_upper: np.ndarray,
        kc_lower: np.ndarray,
        kc_middle: np.ndarray
    ) -> np.ndarray:
        """Map Keltner Channel position to discrete signal."""
        n = len(close)
        result = np.zeros(n, dtype=np.int8)
        
        channel_width = kc_upper - kc_lower
        valid = channel_width > 0
        position = np.zeros(n)
        position[valid] = (close[valid] - kc_lower[valid]) / channel_width[valid]
        
        t = self._thresholds.get('KELTNER', {})
        lower_zone = t.get('lower_zone', 0.25)
        upper_zone = t.get('upper_zone', 0.75)
        
        result[position < 0] = 2
        result[(position >= 0) & (position < lower_zone)] = 1
        result[position > 1.0] = -2
        result[(position > upper_zone) & (position <= 1.0)] = -1
        
        nan_mask = np.isnan(kc_upper) | np.isnan(kc_lower) | np.isnan(kc_middle) | ~valid
        result[nan_mask] = 0
        return result
    
    def atr_signal(self, atr_14: np.ndarray, atr_ma: np.ndarray) -> np.ndarray:
        """Map ATR volatility to discrete signal."""
        n = len(atr_14)
        result = np.zeros(n, dtype=np.int8)
        
        valid = (atr_ma > 0) & ~np.isnan(atr_14) & ~np.isnan(atr_ma)
        ratio = np.ones(n, dtype=np.float64)
        ratio[valid] = atr_14[valid] / atr_ma[valid]
        
        t = self._thresholds.get('ATR', {})
        ext_exp = t.get('extreme_expansion', 2.0)
        exp = t.get('expansion', 1.5)
        ext_sq = t.get('extreme_squeeze', 0.5)
        cont = t.get('contraction', 0.75)
        
        result[ratio > ext_exp] = 2
        result[(ratio > exp) & (ratio <= ext_exp)] = 1
        result[ratio < ext_sq] = -2
        result[(ratio >= ext_sq) & (ratio < cont)] = -1
        
        result[~valid] = 0
        return result
    
    # =========================================================================
    # CONF table discrete signals
    # =========================================================================
    
    def adx_signal(
        self,
        adx: np.ndarray,
        plus_di: np.ndarray,
        minus_di: np.ndarray
    ) -> np.ndarray:
        """Map ADX/DI to discrete signal."""
        n = len(adx)
        result = np.zeros(n, dtype=np.int8)
        
        t = self._thresholds.get('ADX', {})
        non_trending = t.get('non_trending', 25.0)
        strong_trend = t.get('strong_trend', 40.0)
        
        strong_trend_mask = adx > strong_trend
        moderate_trend_mask = (adx >= non_trending) & (adx <= strong_trend)
        bullish = plus_di > minus_di
        bearish = minus_di > plus_di
        
        result[strong_trend_mask & bullish] = 2
        result[strong_trend_mask & bearish] = -2
        result[moderate_trend_mask & bullish] = 1
        result[moderate_trend_mask & bearish] = -1
        
        nan_mask = np.isnan(adx) | np.isnan(plus_di) | np.isnan(minus_di)
        result[nan_mask] = 0
        return result
    
    # =========================================================================
    # Main entry points - per table
    # =========================================================================
    
    def calculate_lead_discrete(
        self,
        ohlcv: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate discrete signals for LEAD table."""
        results = {}
        if 'rsi_14' in indicators:
            results['rsi_signal'] = self.rsi_signal(indicators['rsi_14'])
        if 'stoch_k' in indicators:
            results['stoch_signal'] = self.stoch_signal(indicators['stoch_k'])
        return results
    
    def calculate_coin_discrete(
        self,
        ohlcv: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate discrete signals for COIN table."""
        close = ohlcv['close']
        results = {}
        if all(k in indicators for k in ['macd_12_26_9', 'macd_12_26_9_signal', 'macd_12_26_9_histogram']):
            results['macd_signal'] = self.macd_signal(
                indicators['macd_12_26_9'],
                indicators['macd_12_26_9_signal'],
                indicators['macd_12_26_9_histogram']
            )
        if 'cmf' in indicators:
            results['cmf_signal'] = self.cmf_signal(indicators['cmf'])
        if all(k in indicators for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            results['bb_signal'] = self.bb_signal(
                close,
                indicators['bb_upper'],
                indicators['bb_lower'],
                indicators['bb_middle']
            )
        if all(k in indicators for k in ['keltner_upper', 'keltner_lower', 'keltner_middle']):
            results['keltner_signal'] = self.keltner_signal(
                close,
                indicators['keltner_upper'],
                indicators['keltner_lower'],
                indicators['keltner_middle']
            )
        if 'atr_14' in indicators and 'atr_ma' in indicators:
            results['atr_signal'] = self.atr_signal(indicators['atr_14'], indicators['atr_ma'])
        return results
    
    def calculate_conf_discrete(
        self,
        ohlcv: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Calculate discrete signals for CONF table."""
        results = {}
        if all(k in indicators for k in ['adx_14', 'plus_di_14', 'minus_di_14']):
            results['adx_signal'] = self.adx_signal(
                indicators['adx_14'],
                indicators['plus_di_14'],
                indicators['minus_di_14']
            )
        return results
    
    def calculate_all(
        self,
        ohlcv: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate all discrete signals organized by table."""
        return {
            'lead': self.calculate_lead_discrete(ohlcv, indicators),
            'coin': self.calculate_coin_discrete(ohlcv, indicators),
            'conf': self.calculate_conf_discrete(ohlcv, indicators),
        }
