"""
Boolean signal calculations.

Signals match exact database column names in:
- kfl.signals_lead (48 boolean signals)
- kfl.signals_coin (39 boolean signals)
- kfl.signals_conf (38 boolean signals)

Values:
  1 = bullish/positive condition met
 -1 = bearish/negative condition met
  0 = neutral/no signal

REASON: Thresholds are loaded from qbn.signal_discretization in the database.
"""

import logging
from typing import Dict, Optional, Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)


class BooleanSignalCalculator:
    """
    Calculates boolean signals from indicator values.
    
    Output column names match exactly with database schema.
    Thresholds are loaded from the database at initialization.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize boolean signal calculator.
        
        Args:
            connection_string: Optional DB connection string to load thresholds
        """
        self._thresholds: Dict[str, Dict[str, float]] = {}
        if connection_string:
            self._load_thresholds(connection_string)
        else:
            logger.warning("No connection_string provided to BooleanSignalCalculator, using hardcoded defaults.")
        
        logger.info("Boolean signal calculator initialized")

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
            logger.info(f"✅ {len(rows)} boolean signal thresholds geladen uit database")
        except Exception as e:
            logger.error(f"❌ Fout bij laden thresholds uit database: {e}")
    
    # =========================================================================
    # Helper functions
    # =========================================================================
    
    def _cross_above(self, a: np.ndarray, b: Union[np.ndarray, float, int]) -> np.ndarray:
        """
        Detect when a crosses above b. Returns 1 at cross, 0 otherwise.
        
        REASON: Accepteert scalar threshold om np.full_like allocaties te voorkomen
        """
        n = len(a)
        result = np.zeros(n, dtype=np.int8)
        if np.isscalar(b):
            result[1:] = ((a[1:] > b) & (a[:-1] <= b)).astype(np.int8)
        else:
            result[1:] = ((a[1:] > b[1:]) & (a[:-1] <= b[:-1])).astype(np.int8)
        return result
    
    def _cross_below(self, a: np.ndarray, b: Union[np.ndarray, float, int]) -> np.ndarray:
        """
        Detect when a crosses below b. Returns 1 at cross, 0 otherwise.
        
        REASON: Accepteert scalar threshold om np.full_like allocaties te voorkomen
        """
        n = len(a)
        result = np.zeros(n, dtype=np.int8)
        if np.isscalar(b):
            result[1:] = ((a[1:] < b) & (a[:-1] >= b)).astype(np.int8)
        else:
            result[1:] = ((a[1:] < b[1:]) & (a[:-1] >= b[:-1])).astype(np.int8)
        return result
    
    def _bullish_divergence(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        lookback: int = 14
    ) -> np.ndarray:
        """
        Detect bullish divergence: price makes lower low, indicator makes higher low.
        Returns 1 when detected, 0 otherwise.
        
        REASON: Vectorized with sliding_window_view voor ~10x speedup vs Python loop
        """
        n = len(price)
        result = np.zeros(n, dtype=np.int8)
        
        if n <= lookback:
            return result
        
        # Create sliding windows: shape (n - lookback, lookback + 1)
        pw = sliding_window_view(price, lookback + 1)
        iw = sliding_window_view(indicator, lookback + 1)
        
        # Valid windows (no NaN)
        valid = (~np.isnan(pw).any(axis=1)) & (~np.isnan(iw).any(axis=1))
        
        # Price near low: current <= 20th percentile of window
        thr20 = np.percentile(pw, 20, axis=1)
        near_low = pw[:, -1] <= thr20
        
        # Indicator higher low: current > min of previous values
        prev_ind_min = np.min(iw[:, :-1], axis=1)
        ind_higher_low = iw[:, -1] > prev_ind_min
        
        hits = valid & near_low & ind_higher_low
        result[lookback:] = hits.astype(np.int8)
        
        return result
    
    def _bearish_divergence(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        lookback: int = 14
    ) -> np.ndarray:
        """
        Detect bearish divergence: price makes higher high, indicator makes lower high.
        Returns 1 when detected, 0 otherwise.
        
        REASON: Vectorized with sliding_window_view voor ~10x speedup vs Python loop
        """
        n = len(price)
        result = np.zeros(n, dtype=np.int8)
        
        if n <= lookback:
            return result
        
        # Create sliding windows: shape (n - lookback, lookback + 1)
        pw = sliding_window_view(price, lookback + 1)
        iw = sliding_window_view(indicator, lookback + 1)
        
        # Valid windows (no NaN)
        valid = (~np.isnan(pw).any(axis=1)) & (~np.isnan(iw).any(axis=1))
        
        # Price near high: current >= 80th percentile of window
        thr80 = np.percentile(pw, 80, axis=1)
        near_high = pw[:, -1] >= thr80
        
        # Indicator lower high: current < max of previous values
        prev_ind_max = np.max(iw[:, :-1], axis=1)
        ind_lower_high = iw[:, -1] < prev_ind_max
        
        hits = valid & near_high & ind_lower_high
        result[lookback:] = hits.astype(np.int8)
        
        return result
    
    def _hidden_bullish_divergence(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        lookback: int = 14
    ) -> np.ndarray:
        """
        Detect hidden bullish divergence: price makes higher low, indicator makes lower low.
        Indicates trend continuation in uptrend.
        
        REASON: Vectorized with sliding_window_view voor ~10x speedup vs Python loop
        """
        n = len(price)
        result = np.zeros(n, dtype=np.int8)
        
        if n <= lookback:
            return result
        
        # Create sliding windows
        pw = sliding_window_view(price, lookback + 1)
        iw = sliding_window_view(indicator, lookback + 1)
        
        # Valid windows (no NaN)
        valid = (~np.isnan(pw).any(axis=1)) & (~np.isnan(iw).any(axis=1))
        
        # Price higher low: current > min of previous
        prev_price_min = np.min(pw[:, :-1], axis=1)
        price_higher_low = pw[:, -1] > prev_price_min
        
        # Indicator lower low: current < min of previous
        prev_ind_min = np.min(iw[:, :-1], axis=1)
        ind_lower_low = iw[:, -1] < prev_ind_min
        
        hits = valid & price_higher_low & ind_lower_low
        result[lookback:] = hits.astype(np.int8)
        
        return result
    
    def _hidden_bearish_divergence(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        lookback: int = 14
    ) -> np.ndarray:
        """
        Detect hidden bearish divergence: price makes lower high, indicator makes higher high.
        Indicates trend continuation in downtrend.
        
        REASON: Vectorized with sliding_window_view voor ~10x speedup vs Python loop
        """
        n = len(price)
        result = np.zeros(n, dtype=np.int8)
        
        if n <= lookback:
            return result
        
        # Create sliding windows
        pw = sliding_window_view(price, lookback + 1)
        iw = sliding_window_view(indicator, lookback + 1)
        
        # Valid windows (no NaN)
        valid = (~np.isnan(pw).any(axis=1)) & (~np.isnan(iw).any(axis=1))
        
        # Price lower high: current < max of previous
        prev_price_max = np.max(pw[:, :-1], axis=1)
        price_lower_high = pw[:, -1] < prev_price_max
        
        # Indicator higher high: current > max of previous
        prev_ind_max = np.max(iw[:, :-1], axis=1)
        ind_higher_high = iw[:, -1] > prev_ind_max
        
        hits = valid & price_lower_high & ind_higher_high
        result[lookback:] = hits.astype(np.int8)
        
        return result
    
    def _safe_get(self, indicators: Dict, key: str) -> Optional[np.ndarray]:
        """Safely get indicator, return None if not found."""
        return indicators.get(key)
    
    def _zeros_like_first(self, indicators: Dict) -> np.ndarray:
        """Get zeros array matching length of first indicator."""
        for v in indicators.values():
            if isinstance(v, np.ndarray):
                return np.zeros(len(v), dtype=np.int8)
        return np.zeros(1, dtype=np.int8)
    
    # =========================================================================
    # LEADING signals (48) - Predictive signals
    # =========================================================================
    
    def calculate_leading_signals(
        self,
        ohlcv: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate leading (predictive) signals matching database columns.
        
        Returns dict with exact column names from kfl.signals_lead.
        """
        close = ohlcv['close']
        n = len(close)
        results = {}
        
        # ---------------------------------------------------------------------
        # Keltner Channel signals (7)
        # ---------------------------------------------------------------------
        kc_upper = self._safe_get(indicators, 'keltner_upper')
        kc_lower = self._safe_get(indicators, 'keltner_lower')
        kc_middle = self._safe_get(indicators, 'keltner_middle')
        
        if kc_upper is not None and kc_lower is not None and kc_middle is not None:
            # Mean reversion: price touching outer bands
            # REASON: condition.astype(np.int8) is efficienter dan np.where(..., 1, 0).astype(np.int8)
            results['kc_mean_reversion_long'] = (close <= kc_lower).astype(np.int8)
            results['kc_mean_reversion_short'] = (close >= kc_upper).astype(np.int8)
            
            # Squeeze: narrow channel
            kc_width = kc_upper - kc_lower
            kc_width_pct = kc_width / kc_middle
            squeeze_threshold = np.nanpercentile(kc_width_pct, 20)
            results['kc_squeeze'] = (kc_width_pct < squeeze_threshold).astype(np.int8)
            
            # Dynamic support/resistance
            results['kc_dynamic_support'] = (
                (close > kc_lower) & (close < kc_middle) & (close <= kc_lower * 1.01)
            ).astype(np.int8)
            results['kc_dynamic_resistance'] = (
                (close < kc_upper) & (close > kc_middle) & (close >= kc_upper * 0.99)
            ).astype(np.int8)
            
            # Pullback to middle
            results['kc_pullback_long'] = (
                (close > kc_middle * 0.99) & (close < kc_middle * 1.01)
            ).astype(np.int8)
            results['kc_pullback_short'] = results['kc_pullback_long'].copy()
        else:
            for col in ['kc_mean_reversion_long', 'kc_mean_reversion_short', 'kc_squeeze',
                       'kc_dynamic_support', 'kc_dynamic_resistance', 'kc_pullback_long', 'kc_pullback_short']:
                results[col] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Bollinger Band signals (2)
        # ---------------------------------------------------------------------
        bb_upper = self._safe_get(indicators, 'bb_upper')
        bb_lower = self._safe_get(indicators, 'bb_lower')
        bb_middle = self._safe_get(indicators, 'bb_middle')
        
        if bb_upper is not None and bb_lower is not None:
            bb_width = bb_upper - bb_lower
            if bb_middle is not None:
                bb_width_pct = bb_width / bb_middle
                squeeze_threshold = np.nanpercentile(bb_width_pct, 20)
                results['bb_squeeze'] = (bb_width_pct < squeeze_threshold).astype(np.int8)
            else:
                results['bb_squeeze'] = np.zeros(n, dtype=np.int8)
            
            results['bb_mean_reversion_long'] = (close <= bb_lower).astype(np.int8)
        else:
            results['bb_squeeze'] = np.zeros(n, dtype=np.int8)
            results['bb_mean_reversion_long'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # RSI signals (6)
        # REASON: Thresholds uit DB met fallback naar legacy waarden
        # ---------------------------------------------------------------------
        rsi = self._safe_get(indicators, 'rsi_14')
        
        if rsi is not None:
            t = self._thresholds.get('RSI', {})
            rsi_oversold = t.get('oversold', 30.0)
            rsi_overbought = t.get('overbought', 70.0)
            rsi_extreme_oversold = t.get('extreme_oversold', 20.0)
            rsi_extreme_overbought = t.get('extreme_overbought', 80.0)
            
            results['rsi_oversold'] = (rsi < rsi_oversold).astype(np.int8)
            results['rsi_overbought'] = (rsi > rsi_overbought).astype(np.int8)
            results['rsi_extreme_oversold'] = (rsi < rsi_extreme_oversold).astype(np.int8)
            results['rsi_extreme_overbought'] = (rsi > rsi_extreme_overbought).astype(np.int8)
            results['rsi_divergence_bullish'] = self._bullish_divergence(close, rsi)
            results['rsi_divergence_bearish'] = self._bearish_divergence(close, rsi)
        else:
            for col in ['rsi_oversold', 'rsi_overbought', 'rsi_extreme_oversold',
                       'rsi_extreme_overbought', 'rsi_divergence_bullish', 'rsi_divergence_bearish']:
                results[col] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # MACD 6_13_4 and 8_24_9 signals (4)
        # ---------------------------------------------------------------------
        for variant in ['6_13_4', '8_24_9']:
            macd = self._safe_get(indicators, f'macd_{variant}')
            macd_signal = self._safe_get(indicators, f'macd_{variant}_signal')
            
            if macd is not None and macd_signal is not None:
                results[f'macd_{variant}_bullish_cross'] = self._cross_above(macd, macd_signal)
                results[f'macd_{variant}_bearish_cross'] = self._cross_below(macd, macd_signal)
            else:
                results[f'macd_{variant}_bullish_cross'] = np.zeros(n, dtype=np.int8)
                results[f'macd_{variant}_bearish_cross'] = np.zeros(n, dtype=np.int8)
        
        # MACD divergences (using standard 12_26_9)
        macd_hist = self._safe_get(indicators, 'macd_12_26_9_histogram')
        if macd_hist is not None:
            results['macd_divergence_bullish'] = self._bullish_divergence(close, macd_hist)
            results['macd_divergence_bearish'] = self._bearish_divergence(close, macd_hist)
        else:
            results['macd_divergence_bullish'] = np.zeros(n, dtype=np.int8)
            results['macd_divergence_bearish'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # OBV signals (2)
        # ---------------------------------------------------------------------
        obv = self._safe_get(indicators, 'obv')
        
        if obv is not None:
            results['obv_bullish_divergence'] = self._bullish_divergence(close, obv)
            results['obv_bearish_divergence'] = self._bearish_divergence(close, obv)
        else:
            results['obv_bullish_divergence'] = np.zeros(n, dtype=np.int8)
            results['obv_bearish_divergence'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Stochastic signals (6)
        # REASON: Thresholds uit DB met fallback naar legacy waarden
        # ---------------------------------------------------------------------
        stoch_k = self._safe_get(indicators, 'stoch_k')
        stoch_d = self._safe_get(indicators, 'stoch_d')
        
        if stoch_k is not None:
            t = self._thresholds.get('STOCHASTIC', {})
            stoch_oversold = t.get('oversold', 20.0)
            stoch_overbought = t.get('overbought', 80.0)
            
            results['stoch_oversold'] = (stoch_k < stoch_oversold).astype(np.int8)
            results['stoch_overbought'] = (stoch_k > stoch_overbought).astype(np.int8)
            results['stoch_divergence_bull'] = self._bullish_divergence(close, stoch_k)
            results['stoch_divergence_bear'] = self._bearish_divergence(close, stoch_k)
            results['stoch_hidden_divergence_bull'] = self._hidden_bullish_divergence(close, stoch_k)
            results['stoch_hidden_divergence_bear'] = self._hidden_bearish_divergence(close, stoch_k)
        else:
            for col in ['stoch_oversold', 'stoch_overbought', 'stoch_divergence_bull',
                       'stoch_divergence_bear', 'stoch_hidden_divergence_bull', 'stoch_hidden_divergence_bear']:
                results[col] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Awesome Oscillator signals (4)
        # ---------------------------------------------------------------------
        ao = self._safe_get(indicators, 'ao')
        
        if ao is not None:
            # Twin peaks pattern
            # Bullish: AO below zero, two peaks (second higher), second peak followed by green bar
            # Bearish: AO above zero, two peaks (second lower), second peak followed by red bar
            ao_incr = np.zeros(n, dtype=bool)
            ao_incr[1:] = ao[1:] > ao[:-1]
            
            results['ao_twin_peaks_bullish'] = np.zeros(n, dtype=np.int8)
            results['ao_twin_peaks_bearish'] = np.zeros(n, dtype=np.int8)
            
            # Saucer pattern
            # Bullish: AO > 0, dip then rise (green-red-green)
            # Bearish: AO < 0, rise then dip (red-green-red)
            results['ao_saucer_bullish'] = np.zeros(n, dtype=np.int8)
            results['ao_saucer_bearish'] = np.zeros(n, dtype=np.int8)
            
            for i in range(3, n):
                # Bullish saucer: AO > 0, decreasing then increasing
                if ao[i] > 0 and ao[i-2] > ao[i-1] and ao[i] > ao[i-1]:
                    results['ao_saucer_bullish'][i] = 1
                # Bearish saucer: AO < 0, increasing then decreasing
                if ao[i] < 0 and ao[i-2] < ao[i-1] and ao[i] < ao[i-1]:
                    results['ao_saucer_bearish'][i] = 1
        else:
            for col in ['ao_twin_peaks_bullish', 'ao_twin_peaks_bearish', 
                       'ao_saucer_bullish', 'ao_saucer_bearish']:
                results[col] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # ADX signals (2)
        # REASON: Thresholds uit DB met fallback naar legacy waarden
        # ---------------------------------------------------------------------
        adx = self._safe_get(indicators, 'adx_14')
        
        if adx is not None:
            t = self._thresholds.get('ADX', {})
            adx_strong_trend = t.get('strong_trend', 40.0)
            
            # Trend exhaustion: ADX > strong_trend and declining
            adx_declining = np.zeros(n, dtype=bool)
            adx_declining[1:] = adx[1:] < adx[:-1]
            results['adx_trend_exhaustion'] = ((adx > adx_strong_trend) & adx_declining).astype(np.int8)
            
            # Peak reversal: ADX peaked above strong_trend and now declining
            results['adx_peak_reversal'] = np.zeros(n, dtype=np.int8)
            for i in range(2, n):
                if adx[i-1] > adx_strong_trend and adx[i-1] > adx[i-2] and adx[i] < adx[i-1]:
                    results['adx_peak_reversal'][i] = 1
        else:
            results['adx_trend_exhaustion'] = np.zeros(n, dtype=np.int8)
            results['adx_peak_reversal'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Ichimoku signals - standard 9_26_52 and 6_20_52 variant (6)
        # ---------------------------------------------------------------------
        # Kumo twist (cloud color change)
        senkou_a = self._safe_get(indicators, 'ichi_senkou_a_9_26_52')
        senkou_b = self._safe_get(indicators, 'ichi_senkou_b_9_26_52')
        
        if senkou_a is not None and senkou_b is not None:
            # Kumo twist: senkou_a crosses senkou_b
            results['ichimoku_kumo_twist_bull'] = self._cross_above(senkou_a, senkou_b)
            results['ichimoku_kumo_twist_bear'] = self._cross_below(senkou_a, senkou_b)
        else:
            results['ichimoku_kumo_twist_bull'] = np.zeros(n, dtype=np.int8)
            results['ichimoku_kumo_twist_bear'] = np.zeros(n, dtype=np.int8)
        
        # Ichimoku 6_20_52 variant
        tenkan_6 = self._safe_get(indicators, 'ichi_tenkan_6_20_52')
        kijun_6 = self._safe_get(indicators, 'ichi_kijun_6_20_52')
        senkou_a_6 = self._safe_get(indicators, 'ichi_senkou_a_6_20_52')
        senkou_b_6 = self._safe_get(indicators, 'ichi_senkou_b_6_20_52')
        
        if tenkan_6 is not None and kijun_6 is not None:
            results['ichi_6_20_52_tk_cross_bull'] = self._cross_above(tenkan_6, kijun_6)
            results['ichi_6_20_52_tk_cross_bear'] = self._cross_below(tenkan_6, kijun_6)
        else:
            results['ichi_6_20_52_tk_cross_bull'] = np.zeros(n, dtype=np.int8)
            results['ichi_6_20_52_tk_cross_bear'] = np.zeros(n, dtype=np.int8)
        
        if senkou_a_6 is not None and senkou_b_6 is not None:
            cloud_top_6 = np.maximum(senkou_a_6, senkou_b_6)
            cloud_bottom_6 = np.minimum(senkou_a_6, senkou_b_6)
            results['ichi_6_20_52_kumo_breakout_long'] = self._cross_above(close, cloud_top_6)
            results['ichi_6_20_52_kumo_breakout_short'] = self._cross_below(close, cloud_bottom_6)
        else:
            results['ichi_6_20_52_kumo_breakout_long'] = np.zeros(n, dtype=np.int8)
            results['ichi_6_20_52_kumo_breakout_short'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # VPVR signals (3) - placeholder, requires volume profile calculation
        # ---------------------------------------------------------------------
        vpvr_hvn_upper = self._safe_get(indicators, 'vpvr_hvn_upper')
        vpvr_hvn_lower = self._safe_get(indicators, 'vpvr_hvn_lower')
        vpvr_poc = self._safe_get(indicators, 'vpvr_poc')
        
        if vpvr_hvn_upper is not None:
            results['vpvr_hvn_resistance'] = (
                (close >= vpvr_hvn_upper * 0.99) & (close <= vpvr_hvn_upper * 1.01)
            ).astype(np.int8)
        else:
            results['vpvr_hvn_resistance'] = np.zeros(n, dtype=np.int8)
        
        if vpvr_hvn_lower is not None:
            results['vpvr_hvn_support'] = (
                (close >= vpvr_hvn_lower * 0.99) & (close <= vpvr_hvn_lower * 1.01)
            ).astype(np.int8)
        else:
            results['vpvr_hvn_support'] = np.zeros(n, dtype=np.int8)
        
        if vpvr_poc is not None:
            results['vpvr_poc_touch'] = (
                (close >= vpvr_poc * 0.995) & (close <= vpvr_poc * 1.005)
            ).astype(np.int8)
        else:
            results['vpvr_poc_touch'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # CMF signals (2)
        # ---------------------------------------------------------------------
        cmf = self._safe_get(indicators, 'cmf')
        
        if cmf is not None:
            results['cmf_divergence_bullish'] = self._bullish_divergence(close, cmf)
            results['cmf_divergence_bearish'] = self._bearish_divergence(close, cmf)
        else:
            results['cmf_divergence_bullish'] = np.zeros(n, dtype=np.int8)
            results['cmf_divergence_bearish'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Mean reversion setup (2)
        # REASON: Hergebruik thresholds van RSI en Stochastic uit DB
        # ---------------------------------------------------------------------
        # Combines multiple oversold/overbought conditions
        if rsi is not None and stoch_k is not None and bb_lower is not None:
            t_rsi = self._thresholds.get('RSI', {})
            t_stoch = self._thresholds.get('STOCHASTIC', {})
            rsi_oversold = t_rsi.get('oversold', 30.0)
            rsi_overbought = t_rsi.get('overbought', 70.0)
            stoch_oversold = t_stoch.get('oversold', 20.0)
            stoch_overbought = t_stoch.get('overbought', 80.0)
            
            results['mean_reversion_setup_long'] = (
                (rsi < rsi_oversold) & (stoch_k < stoch_oversold) & (close <= bb_lower)
            ).astype(np.int8)
            results['mean_reversion_setup_short'] = (
                (rsi > rsi_overbought) & (stoch_k > stoch_overbought) & (close >= bb_upper)
            ).astype(np.int8)
        else:
            results['mean_reversion_setup_long'] = np.zeros(n, dtype=np.int8)
            results['mean_reversion_setup_short'] = np.zeros(n, dtype=np.int8)
        
        logger.debug(f"Calculated {len(results)} leading signals")
        return results
    
    # =========================================================================
    # COINCIDENT signals (39) - Current state signals
    # =========================================================================
    
    def calculate_coincident_signals(
        self,
        ohlcv: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate coincident (current state) signals matching database columns.
        
        Returns dict with exact column names from kfl.signals_coin.
        """
        close = ohlcv['close']
        volume = ohlcv.get('volume', np.zeros_like(close))
        n = len(close)
        results = {}
        
        # ---------------------------------------------------------------------
        # CMF signals (4)
        # REASON: Thresholds uit DB met fallback naar legacy waarden
        # ---------------------------------------------------------------------
        cmf = self._safe_get(indicators, 'cmf')
        
        if cmf is not None:
            t = self._thresholds.get('CMF', {})
            cmf_strong_buying = t.get('strong_buying', 0.15)
            cmf_strong_selling = t.get('strong_selling', -0.15)
            
            results['cmf_bullish_bias'] = (cmf > 0).astype(np.int8)
            results['cmf_bearish_bias'] = (cmf < 0).astype(np.int8)
            results['cmf_strong_buying'] = (cmf > cmf_strong_buying).astype(np.int8)
            results['cmf_strong_selling'] = (cmf < cmf_strong_selling).astype(np.int8)
        else:
            for col in ['cmf_bullish_bias', 'cmf_bearish_bias', 'cmf_strong_buying', 'cmf_strong_selling']:
                results[col] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # BB signals (2)
        # ---------------------------------------------------------------------
        bb_upper = self._safe_get(indicators, 'bb_upper')
        bb_lower = self._safe_get(indicators, 'bb_lower')
        
        if bb_upper is not None and bb_lower is not None:
            results['bb_breakout_long'] = self._cross_above(close, bb_upper)
            results['bb_breakout_short'] = self._cross_below(close, bb_lower)
        else:
            results['bb_breakout_long'] = np.zeros(n, dtype=np.int8)
            results['bb_breakout_short'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # MACD signals (4) - standard 12_26_9
        # ---------------------------------------------------------------------
        macd = self._safe_get(indicators, 'macd_12_26_9')
        macd_signal_line = self._safe_get(indicators, 'macd_12_26_9_signal')
        macd_hist = self._safe_get(indicators, 'macd_12_26_9_histogram')
        
        if macd is not None and macd_signal_line is not None:
            results['macd_bullish_cross'] = self._cross_above(macd, macd_signal_line)
            results['macd_bearish_cross'] = self._cross_below(macd, macd_signal_line)
        else:
            results['macd_bullish_cross'] = np.zeros(n, dtype=np.int8)
            results['macd_bearish_cross'] = np.zeros(n, dtype=np.int8)
        
        if macd_hist is not None:
            results['macd_histogram_positive'] = (macd_hist > 0).astype(np.int8)
            results['macd_histogram_negative'] = (macd_hist < 0).astype(np.int8)
        else:
            results['macd_histogram_positive'] = np.zeros(n, dtype=np.int8)
            results['macd_histogram_negative'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # ATR signals (2)
        # ---------------------------------------------------------------------
        atr = self._safe_get(indicators, 'atr_14')
        
        if atr is not None:
            atr_pct = (atr / close) * 100
            results['atr_high_volatility'] = (atr_pct > np.nanpercentile(atr_pct, 80)).astype(np.int8)
            results['atr_low_volatility'] = (atr_pct < np.nanpercentile(atr_pct, 20)).astype(np.int8)
        else:
            results['atr_high_volatility'] = np.zeros(n, dtype=np.int8)
            results['atr_low_volatility'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Stochastic signals (1)
        # ---------------------------------------------------------------------
        stoch_k = self._safe_get(indicators, 'stoch_k')
        stoch_d = self._safe_get(indicators, 'stoch_d')
        
        if stoch_k is not None and stoch_d is not None:
            results['stoch_bullish_cross'] = self._cross_above(stoch_k, stoch_d)
        else:
            results['stoch_bullish_cross'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # AO signals (2)
        # ---------------------------------------------------------------------
        ao = self._safe_get(indicators, 'ao')
        
        if ao is not None:
            # REASON: scalar threshold voorkomt np.zeros_like allocatie
            results['ao_bullish_zero_cross'] = self._cross_above(ao, 0.0)
            results['ao_bearish_zero_cross'] = self._cross_below(ao, 0.0)
        else:
            results['ao_bullish_zero_cross'] = np.zeros(n, dtype=np.int8)
            results['ao_bearish_zero_cross'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Volatility breakout (2)
        # ---------------------------------------------------------------------
        if atr is not None and bb_upper is not None:
            # Volatility expansion with price breakout
            atr_expanding = np.zeros(n, dtype=bool)
            atr_expanding[1:] = atr[1:] > atr[:-1]
            
            results['volatility_breakout_long'] = (atr_expanding & (close > bb_upper)).astype(np.int8)
            results['volatility_breakout_short'] = (atr_expanding & (close < bb_lower)).astype(np.int8)
        else:
            results['volatility_breakout_long'] = np.zeros(n, dtype=np.int8)
            results['volatility_breakout_short'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Confluence signals (4)
        # ---------------------------------------------------------------------
        rsi = self._safe_get(indicators, 'rsi_14')
        
        # Momentum confluence: RSI + MACD aligned
        if rsi is not None and macd is not None and macd_signal_line is not None:
            results['momentum_bullish_confluence'] = (
                (rsi > 50) & (macd > macd_signal_line) & (macd > 0)
            ).astype(np.int8)
            results['momentum_bearish_confluence'] = (
                (rsi < 50) & (macd < macd_signal_line) & (macd < 0)
            ).astype(np.int8)
        else:
            results['momentum_bullish_confluence'] = np.zeros(n, dtype=np.int8)
            results['momentum_bearish_confluence'] = np.zeros(n, dtype=np.int8)
        
        # Volume confluence: CMF + OBV trend aligned
        obv = self._safe_get(indicators, 'obv')
        if cmf is not None and obv is not None:
            obv_rising = np.zeros(n, dtype=bool)
            obv_rising[1:] = obv[1:] > obv[:-1]
            
            results['volume_bullish_confluence'] = ((cmf > 0) & obv_rising).astype(np.int8)
            results['volume_bearish_confluence'] = ((cmf < 0) & ~obv_rising).astype(np.int8)
        else:
            results['volume_bullish_confluence'] = np.zeros(n, dtype=np.int8)
            results['volume_bearish_confluence'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Risk signals (3)
        # ---------------------------------------------------------------------
        results['risk_high_volatility'] = results.get('atr_high_volatility', np.zeros(n, dtype=np.int8))
        
        # Low liquidity proxy: volume below 20th percentile
        if volume is not None and np.any(volume > 0):
            vol_threshold = np.nanpercentile(volume, 20)
            results['risk_low_liquidity'] = (volume < vol_threshold).astype(np.int8)
        else:
            results['risk_low_liquidity'] = np.zeros(n, dtype=np.int8)
        
        # Regime volatile
        if atr is not None:
            atr_pct = (atr / close) * 100
            results['regime_volatile'] = (atr_pct > np.nanpercentile(atr_pct, 70)).astype(np.int8)
        else:
            results['regime_volatile'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # DI signals (2)
        # ---------------------------------------------------------------------
        plus_di = self._safe_get(indicators, 'plus_di_14')
        minus_di = self._safe_get(indicators, 'minus_di_14')
        
        if plus_di is not None and minus_di is not None:
            results['di_bullish_cross'] = self._cross_above(plus_di, minus_di)
            results['di_bearish_cross'] = self._cross_below(plus_di, minus_di)
        else:
            results['di_bullish_cross'] = np.zeros(n, dtype=np.int8)
            results['di_bearish_cross'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Supertrend signals (2)
        # ---------------------------------------------------------------------
        st_dir = self._safe_get(indicators, 'supertrend_dir_10_3')
        
        if st_dir is not None:
            # Flip detection
            results['super_trend_flip_bull'] = np.zeros(n, dtype=np.int8)
            results['super_trend_flip_bear'] = np.zeros(n, dtype=np.int8)
            results['super_trend_flip_bull'][1:] = np.where(
                (st_dir[1:] > 0) & (st_dir[:-1] < 0), 1, 0
            )
            results['super_trend_flip_bear'][1:] = np.where(
                (st_dir[1:] < 0) & (st_dir[:-1] > 0), 1, 0
            )
        else:
            results['super_trend_flip_bull'] = np.zeros(n, dtype=np.int8)
            results['super_trend_flip_bear'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Ichimoku signals - standard 9_26_52 (5)
        # ---------------------------------------------------------------------
        tenkan = self._safe_get(indicators, 'ichi_tenkan_9_26_52')
        kijun = self._safe_get(indicators, 'ichi_kijun_9_26_52')
        senkou_a = self._safe_get(indicators, 'ichi_senkou_a_9_26_52')
        senkou_b = self._safe_get(indicators, 'ichi_senkou_b_9_26_52')
        
        if tenkan is not None and kijun is not None:
            results['ichimoku_tenkan_kijun_cross_bull'] = self._cross_above(tenkan, kijun)
            results['ichimoku_tenkan_kijun_cross_bear'] = self._cross_below(tenkan, kijun)
        else:
            results['ichimoku_tenkan_kijun_cross_bull'] = np.zeros(n, dtype=np.int8)
            results['ichimoku_tenkan_kijun_cross_bear'] = np.zeros(n, dtype=np.int8)
        
        if senkou_a is not None and senkou_b is not None:
            cloud_top = np.maximum(senkou_a, senkou_b)
            cloud_bottom = np.minimum(senkou_a, senkou_b)
            
            results['ichimoku_price_above_kumo'] = (close > cloud_top).astype(np.int8)
            results['ichimoku_price_below_kumo'] = (close < cloud_bottom).astype(np.int8)
            results['ichimoku_price_in_kumo'] = (
                (close >= cloud_bottom) & (close <= cloud_top)
            ).astype(np.int8)
        else:
            results['ichimoku_price_above_kumo'] = np.zeros(n, dtype=np.int8)
            results['ichimoku_price_below_kumo'] = np.zeros(n, dtype=np.int8)
            results['ichimoku_price_in_kumo'] = np.zeros(n, dtype=np.int8)
        
        # Ichimoku 7_22_44 variant (2)
        tenkan_7 = self._safe_get(indicators, 'ichi_tenkan_7_22_44')
        kijun_7 = self._safe_get(indicators, 'ichi_kijun_7_22_44')
        
        if tenkan_7 is not None and kijun_7 is not None:
            results['ichi_7_22_44_tk_cross_bull'] = self._cross_above(tenkan_7, kijun_7)
            results['ichi_7_22_44_tk_cross_bear'] = self._cross_below(tenkan_7, kijun_7)
        else:
            results['ichi_7_22_44_tk_cross_bull'] = np.zeros(n, dtype=np.int8)
            results['ichi_7_22_44_tk_cross_bear'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # VPVR signals (2) - LVN breakouts
        # ---------------------------------------------------------------------
        vpvr_lvn_upper = self._safe_get(indicators, 'vpvr_lvn_upper')
        vpvr_lvn_lower = self._safe_get(indicators, 'vpvr_lvn_lower')
        
        if vpvr_lvn_upper is not None:
            results['vpvr_lvn_breakout_up'] = self._cross_above(close, vpvr_lvn_upper)
        else:
            results['vpvr_lvn_breakout_up'] = np.zeros(n, dtype=np.int8)
        
        if vpvr_lvn_lower is not None:
            results['vpvr_lvn_breakout_down'] = self._cross_below(close, vpvr_lvn_lower)
        else:
            results['vpvr_lvn_breakout_down'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # RSI center cross (2)
        # ---------------------------------------------------------------------
        if rsi is not None:
            # REASON: scalar threshold voorkomt np.full_like allocatie
            results['rsi_center_cross_bull'] = self._cross_above(rsi, 50.0)
            results['rsi_center_cross_bear'] = self._cross_below(rsi, 50.0)
        else:
            results['rsi_center_cross_bull'] = np.zeros(n, dtype=np.int8)
            results['rsi_center_cross_bear'] = np.zeros(n, dtype=np.int8)
        
        logger.debug(f"Calculated {len(results)} coincident signals")
        return results
    
    # =========================================================================
    # CONFIRMING signals (38) - Trend confirmation signals
    # =========================================================================
    
    def calculate_confirming_signals(
        self,
        ohlcv: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate confirming (trend confirmation) signals matching database columns.
        
        Returns dict with exact column names from kfl.signals_conf.
        """
        close = ohlcv['close']
        n = len(close)
        results = {}
        
        # ---------------------------------------------------------------------
        # ADX signals (4)
        # REASON: Thresholds uit DB met fallback naar legacy waarden
        # ---------------------------------------------------------------------
        adx = self._safe_get(indicators, 'adx_14')
        plus_di = self._safe_get(indicators, 'plus_di_14')
        minus_di = self._safe_get(indicators, 'minus_di_14')
        
        if adx is not None:
            t = self._thresholds.get('ADX', {})
            adx_non_trending = t.get('non_trending', 25.0)
            adx_strong_trend = t.get('strong_trend', 40.0)
            # REASON: weak_trend zone = tussen non_trending-5 en non_trending
            adx_weak_lower = adx_non_trending - 5.0  # 20 als non_trending=25
            
            results['adx_trend_confirm'] = (adx > adx_non_trending).astype(np.int8)
            results['adx_non_trending_regime'] = (adx < adx_weak_lower).astype(np.int8)
            results['adx_strong_trend'] = (adx > adx_strong_trend).astype(np.int8)
            results['adx_weak_trend'] = ((adx >= adx_weak_lower) & (adx <= adx_non_trending)).astype(np.int8)
        else:
            for col in ['adx_trend_confirm', 'adx_non_trending_regime', 'adx_strong_trend', 'adx_weak_trend']:
                results[col] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Keltner Channel signals (2)
        # ---------------------------------------------------------------------
        kc_upper = self._safe_get(indicators, 'keltner_upper')
        kc_lower = self._safe_get(indicators, 'keltner_lower')
        
        if kc_upper is not None and kc_lower is not None:
            results['kc_trend_breakout_long'] = self._cross_above(close, kc_upper)
            results['kc_trend_breakout_short'] = self._cross_below(close, kc_lower)
        else:
            results['kc_trend_breakout_long'] = np.zeros(n, dtype=np.int8)
            results['kc_trend_breakout_short'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # MACD 5_35_5 and 20_50_15 signals (4)
        # ---------------------------------------------------------------------
        for variant in ['5_35_5', '20_50_15']:
            macd = self._safe_get(indicators, f'macd_{variant}')
            macd_signal = self._safe_get(indicators, f'macd_{variant}_signal')
            
            if macd is not None and macd_signal is not None:
                results[f'macd_{variant}_bullish_cross'] = self._cross_above(macd, macd_signal)
                results[f'macd_{variant}_bearish_cross'] = self._cross_below(macd, macd_signal)
            else:
                results[f'macd_{variant}_bullish_cross'] = np.zeros(n, dtype=np.int8)
                results[f'macd_{variant}_bearish_cross'] = np.zeros(n, dtype=np.int8)
        
        # MACD zero line cross (using standard 12_26_9)
        macd_std = self._safe_get(indicators, 'macd_12_26_9')
        if macd_std is not None:
            # REASON: scalar threshold voorkomt np.zeros_like allocatie
            results['macd_zero_line_cross_bull'] = self._cross_above(macd_std, 0.0)
            results['macd_zero_line_cross_bear'] = self._cross_below(macd_std, 0.0)
        else:
            results['macd_zero_line_cross_bull'] = np.zeros(n, dtype=np.int8)
            results['macd_zero_line_cross_bear'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Confluence signals (2)
        # REASON: Hergebruik ADX threshold uit DB
        # ---------------------------------------------------------------------
        rsi = self._safe_get(indicators, 'rsi_14')
        ema_50 = self._safe_get(indicators, 'ema_50')
        ema_200 = self._safe_get(indicators, 'ema_200')
        
        # Strong confluence: ADX trending + RSI confirming + price above/below EMAs
        if adx is not None and rsi is not None and ema_50 is not None:
            results['bullish_confluence_strong'] = (
                (adx > adx_non_trending) & (rsi > 50) & (close > ema_50)
            ).astype(np.int8)
            results['bearish_confluence_strong'] = (
                (adx > adx_non_trending) & (rsi < 50) & (close < ema_50)
            ).astype(np.int8)
        else:
            results['bullish_confluence_strong'] = np.zeros(n, dtype=np.int8)
            results['bearish_confluence_strong'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Trend following breakout (2)
        # REASON: Hergebruik ADX threshold uit DB
        # ---------------------------------------------------------------------
        if adx is not None and ema_50 is not None:
            results['trend_following_breakout_long'] = ((adx > adx_non_trending) & (close > ema_50)).astype(np.int8)
            results['trend_following_breakout_short'] = ((adx > adx_non_trending) & (close < ema_50)).astype(np.int8)
        else:
            results['trend_following_breakout_long'] = np.zeros(n, dtype=np.int8)
            results['trend_following_breakout_short'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # MTF alignment (2) - using different MA lengths as proxy
        # ---------------------------------------------------------------------
        ema_7 = self._safe_get(indicators, 'ema_7')
        ema_25 = self._safe_get(indicators, 'ema_25')
        
        if ema_7 is not None and ema_25 is not None and ema_50 is not None:
            results['mtf_bullish_alignment'] = ((ema_7 > ema_25) & (ema_25 > ema_50)).astype(np.int8)
            results['mtf_bearish_alignment'] = ((ema_7 < ema_25) & (ema_25 < ema_50)).astype(np.int8)
        else:
            results['mtf_bullish_alignment'] = np.zeros(n, dtype=np.int8)
            results['mtf_bearish_alignment'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Regime signals (3)
        # REASON: Hergebruik ADX thresholds uit DB
        # ---------------------------------------------------------------------
        if adx is not None and plus_di is not None and minus_di is not None:
            results['regime_trending_bullish'] = ((adx > adx_non_trending) & (plus_di > minus_di)).astype(np.int8)
            results['regime_trending_bearish'] = ((adx > adx_non_trending) & (minus_di > plus_di)).astype(np.int8)
            results['regime_ranging'] = (adx < adx_weak_lower).astype(np.int8)
        else:
            results['regime_trending_bullish'] = np.zeros(n, dtype=np.int8)
            results['regime_trending_bearish'] = np.zeros(n, dtype=np.int8)
            results['regime_ranging'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # DI signals (2)
        # REASON: Hergebruik ADX threshold uit DB
        # ---------------------------------------------------------------------
        if plus_di is not None and minus_di is not None and adx is not None:
            results['di_strong_bullish'] = ((plus_di > minus_di) & (adx > adx_non_trending)).astype(np.int8)
            results['di_strong_bearish'] = ((minus_di > plus_di) & (adx > adx_non_trending)).astype(np.int8)
        else:
            results['di_strong_bullish'] = np.zeros(n, dtype=np.int8)
            results['di_strong_bearish'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Supertrend signals (2)
        # ---------------------------------------------------------------------
        st_dir = self._safe_get(indicators, 'supertrend_dir_10_3')
        
        if st_dir is not None:
            results['super_trend_bullish'] = (st_dir > 0).astype(np.int8)
            results['super_trend_bearish'] = (st_dir < 0).astype(np.int8)
        else:
            results['super_trend_bullish'] = np.zeros(n, dtype=np.int8)
            results['super_trend_bearish'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # Ichimoku signals - standard 9_26_52 and 10_30_60 variant (6)
        # ---------------------------------------------------------------------
        senkou_a = self._safe_get(indicators, 'ichi_senkou_a_9_26_52')
        senkou_b = self._safe_get(indicators, 'ichi_senkou_b_9_26_52')
        
        if senkou_a is not None and senkou_b is not None:
            cloud_top = np.maximum(senkou_a, senkou_b)
            cloud_bottom = np.minimum(senkou_a, senkou_b)
            results['ichimoku_kumo_breakout_long'] = self._cross_above(close, cloud_top)
            results['ichimoku_kumo_breakout_short'] = self._cross_below(close, cloud_bottom)
        else:
            results['ichimoku_kumo_breakout_long'] = np.zeros(n, dtype=np.int8)
            results['ichimoku_kumo_breakout_short'] = np.zeros(n, dtype=np.int8)
        
        # Ichimoku 10_30_60 variant
        tenkan_10 = self._safe_get(indicators, 'ichi_tenkan_10_30_60')
        kijun_10 = self._safe_get(indicators, 'ichi_kijun_10_30_60')
        senkou_a_10 = self._safe_get(indicators, 'ichi_senkou_a_10_30_60')
        senkou_b_10 = self._safe_get(indicators, 'ichi_senkou_b_10_30_60')
        
        if tenkan_10 is not None and kijun_10 is not None:
            results['ichi_10_30_60_tk_cross_bull'] = self._cross_above(tenkan_10, kijun_10)
            results['ichi_10_30_60_tk_cross_bear'] = self._cross_below(tenkan_10, kijun_10)
        else:
            results['ichi_10_30_60_tk_cross_bull'] = np.zeros(n, dtype=np.int8)
            results['ichi_10_30_60_tk_cross_bear'] = np.zeros(n, dtype=np.int8)
        
        if senkou_a_10 is not None and senkou_b_10 is not None:
            cloud_top_10 = np.maximum(senkou_a_10, senkou_b_10)
            cloud_bottom_10 = np.minimum(senkou_a_10, senkou_b_10)
            results['ichi_10_30_60_kumo_breakout_long'] = self._cross_above(close, cloud_top_10)
            results['ichi_10_30_60_kumo_breakout_short'] = self._cross_below(close, cloud_bottom_10)
        else:
            results['ichi_10_30_60_kumo_breakout_long'] = np.zeros(n, dtype=np.int8)
            results['ichi_10_30_60_kumo_breakout_short'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # VPVR signals (1)
        # ---------------------------------------------------------------------
        vpvr_vah = self._safe_get(indicators, 'vpvr_vah')
        vpvr_val = self._safe_get(indicators, 'vpvr_val')
        
        if vpvr_vah is not None and vpvr_val is not None:
            results['vpvr_value_area_inside'] = ((close >= vpvr_val) & (close <= vpvr_vah)).astype(np.int8)
        else:
            results['vpvr_value_area_inside'] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # OBV signals (4)
        # ---------------------------------------------------------------------
        obv = self._safe_get(indicators, 'obv')
        
        if obv is not None:
            obv_rising = np.zeros(n, dtype=bool)
            obv_rising[1:] = obv[1:] > obv[:-1]
            
            price_rising = np.zeros(n, dtype=bool)
            price_rising[1:] = close[1:] > close[:-1]
            
            # Trend confirm: OBV and price moving same direction
            results['obv_trend_confirm_bull'] = (obv_rising & price_rising).astype(np.int8)
            results['obv_trend_confirm_bear'] = (~obv_rising & ~price_rising).astype(np.int8)
            
            # Trend strength: strong OBV momentum
            obv_change = np.zeros(n)
            obv_change[1:] = obv[1:] - obv[:-1]
            obv_strong_up = obv_change > np.nanpercentile(obv_change, 80)
            obv_strong_down = obv_change < np.nanpercentile(obv_change, 20)
            
            results['obv_trend_strength_bull'] = obv_strong_up.astype(np.int8)
            results['obv_trend_strength_bear'] = obv_strong_down.astype(np.int8)
        else:
            for col in ['obv_trend_confirm_bull', 'obv_trend_confirm_bear',
                       'obv_trend_strength_bull', 'obv_trend_strength_bear']:
                results[col] = np.zeros(n, dtype=np.int8)
        
        # ---------------------------------------------------------------------
        # RSI center signals (2)
        # ---------------------------------------------------------------------
        if rsi is not None:
            results['rsi_center_bullish'] = (rsi > 50).astype(np.int8)
            results['rsi_center_bearish'] = (rsi < 50).astype(np.int8)
        else:
            results['rsi_center_bullish'] = np.zeros(n, dtype=np.int8)
            results['rsi_center_bearish'] = np.zeros(n, dtype=np.int8)
        
        logger.debug(f"Calculated {len(results)} confirming signals")
        return results
    
    # =========================================================================
    # Main entry point
    # =========================================================================
    
    def calculate_all(
        self,
        ohlcv: Dict[str, np.ndarray],
        indicators: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate all boolean signals organized by classification.
        
        Args:
            ohlcv: Dict with keys: open, high, low, close, volume
            indicators: Dict with all calculated indicator values
            
        Returns:
            Dict with keys: 'leading', 'coincident', 'confirming'
            Each value is a dict of signal_name -> signal_array
        """
        return {
            'leading': self.calculate_leading_signals(ohlcv, indicators),
            'coincident': self.calculate_coincident_signals(ohlcv, indicators),
            'confirming': self.calculate_confirming_signals(ohlcv, indicators),
        }
