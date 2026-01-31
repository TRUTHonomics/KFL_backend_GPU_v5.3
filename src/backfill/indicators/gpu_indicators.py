"""
GPU-accelerated indicator calculations using CuPy.

Vectorized indicators that benefit from GPU parallelization:
- SMA (Simple Moving Average)
- Bollinger Bands
- Stochastic Oscillator
- Ichimoku Cloud
- OBV (On Balance Volume)
- CMF (Chaikin Money Flow)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

logger = logging.getLogger(__name__)


class GPUIndicatorCalculator:
    """
    GPU-accelerated indicator calculations using CuPy.
    
    All methods accept numpy arrays and return numpy arrays.
    Internally uses CuPy for GPU computation when available.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU calculator.
        
        Args:
            use_gpu: Whether to use GPU (falls back to CPU if unavailable)
        """
        self.use_gpu = use_gpu and HAS_CUPY
        if self.use_gpu:
            logger.info("GPU indicator calculator initialized with CuPy")
        else:
            logger.info("GPU indicator calculator using CPU fallback")
    
    def _to_device(self, arr: np.ndarray) -> 'cp.ndarray | np.ndarray':
        """Transfer array to GPU if available."""
        if self.use_gpu:
            return cp.asarray(arr)
        return arr
    
    def _to_host(self, arr: 'cp.ndarray | np.ndarray') -> np.ndarray:
        """Transfer array back to CPU."""
        if self.use_gpu and hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
    
    def _get_xp(self):
        """Get the array library (cupy or numpy)."""
        return cp if self.use_gpu else np

    def _as_f64(self, arr: np.ndarray) -> np.ndarray:
        """
        Cast to float64 without copy if already float64.

        Uses np.asarray which avoids unnecessary copy when dtype matches.
        """
        return np.asarray(arr, dtype=np.float64)

    def _to_device_f64(self, arr: np.ndarray) -> 'cp.ndarray | np.ndarray':
        """
        Cast to float64 and transfer to device in one operation.

        Optimizes the common pattern of astype().then._to_device().
        """
        return self._to_device(self._as_f64(arr))

    def _rolling_max_min_gpu(
        self,
        arr: 'cp.ndarray | np.ndarray',
        window: int
    ) -> Tuple['cp.ndarray | np.ndarray', 'cp.ndarray | np.ndarray']:
        """
        Calculate rolling max and min using GPU primitives.

        Uses cupyx.scipy.ndimage filters for true GPU acceleration.
        Falls back to simple loop implementation if cupyx.scipy not available.

        CRITICAL OPTIMIZATION: Replaces Python for-loops with single GPU kernel calls.

        Args:
            arr: Input array (already on device)
            window: Rolling window size

        Returns:
            Tuple of (rolling_max, rolling_min)
        """
        xp = self._get_xp()
        n = len(arr)

        if not self.use_gpu:
            # CPU fallback - simple rolling window
            rolling_max = xp.full(n, xp.nan, dtype=xp.float64)
            rolling_min = xp.full(n, xp.nan, dtype=xp.float64)
            for i in range(window - 1, n):
                window_slice = arr[i - window + 1:i + 1]
                rolling_max[i] = xp.max(window_slice)
                rolling_min[i] = xp.min(window_slice)
            return rolling_max, rolling_min

        # GPU: Use cupyx.scipy.ndimage filters for massive speedup
        try:
            from cupyx.scipy import ndimage

            # Mode 'constant' with cval=inf ensures proper edge handling
            # Size = window for 1D filter
            rolling_max = ndimage.maximum_filter1d(
                arr, size=window, mode='constant', cval=-xp.inf
            )
            rolling_min = ndimage.minimum_filter1d(
                arr, size=window, mode='constant', cval=xp.inf
            )

            # Set first window-1 values to NaN (not enough data for full window)
            rolling_max[:window-1] = xp.nan
            rolling_min[:window-1] = xp.nan

            return rolling_max, rolling_min

        except ImportError:
            logger.warning(
                "cupyx.scipy not available - using slow GPU fallback for rolling operations. "
                "Install cupyx for optimal performance."
            )
            # Fallback to current loop-based implementation
            rolling_max = xp.full(n, xp.nan, dtype=xp.float64)
            rolling_min = xp.full(n, xp.nan, dtype=xp.float64)
            for i in range(window - 1, n):
                window_slice = arr[i - window + 1:i + 1]
                rolling_max[i] = xp.max(window_slice)
                rolling_min[i] = xp.min(window_slice)
            return rolling_max, rolling_min

    # =========================================================================
    # SMA - Simple Moving Average
    # =========================================================================
    
    def sma(self, close: np.ndarray, period: int, *, _device: bool = False) -> np.ndarray:
        """
        Calculate Simple Moving Average.

        Uses cumsum trick for O(n) computation regardless of period.

        Args:
            close: Close prices
            period: SMA period
            _device: If True, assumes close is already on device as float64

        Returns:
            SMA values (first period-1 values are NaN)
        """
        xp = self._get_xp()
        close_d = close if _device else self._to_device_f64(close)
        
        n = len(close_d)
        result = xp.full(n, xp.nan, dtype=xp.float64)
        
        if n < period:
            return self._to_host(result)
        
        # Cumsum approach for vectorized rolling mean
        cumsum = xp.cumsum(close_d)
        result[period-1:] = (cumsum[period-1:] - xp.concatenate([xp.array([0.0]), cumsum[:-period]])) / period
        
        return self._to_host(result)
    
    def calculate_all_smas(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate all SMA variants used in signals.
        
        Returns:
            Dict with keys: sma_7, sma_25, sma_50, sma_99, sma_200
        """
        periods = {'sma_7': 7, 'sma_25': 25, 'sma_50': 50, 'sma_99': 99, 'sma_200': 200}
        return {name: self.sma(close, period) for name, period in periods.items()}
    
    # =========================================================================
    # Bollinger Bands
    # =========================================================================
    
    def bollinger_bands(
        self,
        close: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0,
        *,
        _device: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            close: Close prices
            period: BB period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            _device: If True, assumes close is already on device as float64

        Returns:
            Tuple of (middle, upper, lower, bandwidth)
        """
        xp = self._get_xp()
        close_d = close if _device else self._to_device_f64(close)
        
        n = len(close_d)
        middle = xp.full(n, xp.nan, dtype=xp.float64)
        upper = xp.full(n, xp.nan, dtype=xp.float64)
        lower = xp.full(n, xp.nan, dtype=xp.float64)
        bandwidth = xp.full(n, xp.nan, dtype=xp.float64)
        
        if n < period:
            return tuple(self._to_host(arr) for arr in [middle, upper, lower, bandwidth])
        
        # Rolling mean using cumsum
        cumsum = xp.cumsum(close_d)
        middle[period-1:] = (cumsum[period-1:] - xp.concatenate([xp.array([0.0]), cumsum[:-period]])) / period
        
        # Rolling std using the formula: std = sqrt(E[X^2] - E[X]^2)
        cumsum_sq = xp.cumsum(close_d ** 2)
        mean_sq = (cumsum_sq[period-1:] - xp.concatenate([xp.array([0.0]), cumsum_sq[:-period]])) / period
        sq_mean = middle[period-1:] ** 2
        std = xp.sqrt(xp.maximum(mean_sq - sq_mean, 0))
        
        upper[period-1:] = middle[period-1:] + std_dev * std
        lower[period-1:] = middle[period-1:] - std_dev * std
        
        # Bandwidth = (upper - lower) / middle * 100
        with np.errstate(divide='ignore', invalid='ignore'):
            bandwidth[period-1:] = xp.where(
                middle[period-1:] != 0,
                (upper[period-1:] - lower[period-1:]) / middle[period-1:] * 100,
                xp.nan
            )
        
        return tuple(self._to_host(arr) for arr in [middle, upper, lower, bandwidth])
    
    def calculate_bollinger_variants(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate all Bollinger Band variants.
        
        Returns:
            Dict with keys: bb_middle, bb_upper, bb_lower, bb_bandwidth
        """
        middle, upper, lower, bandwidth = self.bollinger_bands(close, period=20, std_dev=2.0)
        return {
            'bb_middle': middle,
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_bandwidth': bandwidth,
        }
    
    # =========================================================================
    # Stochastic Oscillator
    # =========================================================================
    
    def stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
        *,
        _device: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator.

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D smoothing period
            _device: If True, assumes inputs are already on device as float64

        Returns:
            Tuple of (stoch_k, stoch_d)
        """
        xp = self._get_xp()
        if _device:
            high_d, low_d, close_d = high, low, close
        else:
            high_d = self._to_device_f64(high)
            low_d = self._to_device_f64(low)
            close_d = self._to_device_f64(close)

        n = len(close_d)
        stoch_k = xp.full(n, xp.nan, dtype=xp.float64)
        stoch_d = xp.full(n, xp.nan, dtype=xp.float64)

        if n < k_period:
            return self._to_host(stoch_k), self._to_host(stoch_d)

        # ===== OPTIMIZATION: Use GPU rolling max/min (1 kernel call per array!) =====
        rolling_high, _ = self._rolling_max_min_gpu(high_d, k_period)
        _, rolling_low = self._rolling_max_min_gpu(low_d, k_period)

        # Calculate %K vectorized (no loops!)
        range_val = rolling_high - rolling_low
        stoch_k = xp.where(
            range_val > 0,
            (close_d - rolling_low) / range_val * 100,
            50.0  # Default when range is zero
        )

        # %D is SMA of %K
        k_valid_start = k_period - 1
        d_start = k_valid_start + d_period - 1

        if d_start < n:
            # Cumsum trick for SMA: prepend 0 for easier indexing
            k_filled = xp.nan_to_num(stoch_k, nan=0.0)
            k_cumsum = xp.concatenate([xp.array([0.0]), xp.cumsum(k_filled)])

            # Vectorized SMA calculation
            # For each position i, SMA = (cumsum[i+1] - cumsum[i+1-d_period]) / d_period
            stoch_d[d_start:] = (
                k_cumsum[d_start+1:n+1] - k_cumsum[d_start+1-d_period:n+1-d_period]
            ) / d_period

        return self._to_host(stoch_k), self._to_host(stoch_d)
    
    def calculate_stochastic_variants(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all Stochastic variants.
        
        Returns:
            Dict with keys: stoch_k, stoch_d
        """
        stoch_k, stoch_d = self.stochastic(high, low, close, k_period=14, d_period=3)
        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
        }
    
    # =========================================================================
    # Ichimoku Cloud
    # =========================================================================
    
    def ichimoku(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
        *,
        _device: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Ichimoku Cloud indicators.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            tenkan_period: Tenkan-sen period (default 9)
            kijun_period: Kijun-sen period (default 26)
            senkou_b_period: Senkou Span B period (default 52)
            displacement: Cloud displacement (default 26)
            _device: If True, assumes inputs are already on device as float64

        Returns:
            Dict with: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        """
        xp = self._get_xp()
        if _device:
            high_d, low_d, close_d = high, low, close
        else:
            high_d = self._to_device_f64(high)
            low_d = self._to_device_f64(low)
            close_d = self._to_device_f64(close)
        
        n = len(close_d)

        # ===== OPTIMIZATION: Vectorized donchian_middle using GPU rolling =====
        def donchian_middle_vectorized(h, l, period):
            """Calculate (highest_high + lowest_low) / 2 over period - OPTIMIZED."""
            rolling_high, _ = self._rolling_max_min_gpu(h, period)
            _, rolling_low = self._rolling_max_min_gpu(l, period)
            return (rolling_high + rolling_low) / 2

        # Tenkan-sen (Conversion Line)
        tenkan_sen = donchian_middle_vectorized(high_d, low_d, tenkan_period)

        # Kijun-sen (Base Line)
        kijun_sen = donchian_middle_vectorized(high_d, low_d, kijun_period)
        
        # Senkou Span A (Leading Span A) = (Tenkan + Kijun) / 2, displaced forward
        senkou_a_base = (tenkan_sen + kijun_sen) / 2
        senkou_span_a = xp.full(n, xp.nan, dtype=xp.float64)
        if displacement < n:
            senkou_span_a[displacement:] = senkou_a_base[:-displacement]
        
        # Senkou Span B (Leading Span B) = Donchian middle of 52 periods, displaced forward
        senkou_b_base = donchian_middle_vectorized(high_d, low_d, senkou_b_period)
        senkou_span_b = xp.full(n, xp.nan, dtype=xp.float64)
        if displacement < n:
            senkou_span_b[displacement:] = senkou_b_base[:-displacement]
        
        # Chikou Span (Lagging Span) = Close displaced backward
        chikou_span = xp.full(n, xp.nan, dtype=xp.float64)
        if displacement < n:
            chikou_span[:-displacement] = close_d[displacement:]
        
        return {
            'tenkan_sen': self._to_host(tenkan_sen),
            'kijun_sen': self._to_host(kijun_sen),
            'senkou_span_a': self._to_host(senkou_span_a),
            'senkou_span_b': self._to_host(senkou_span_b),
            'chikou_span': self._to_host(chikou_span),
        }
    
    def calculate_ichimoku_variants(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all Ichimoku variants.
        
        Variants:
        - 9_26_52: Standard Ichimoku (for traditional markets)
        - 10_30_60: Crypto parameters (24/7 markets) - used in CONF signals
        - 7_22_44: Fast variant - used in COIN signals
        - 6_20_52: Alternative fast variant - used in LEAD signals
        
        Returns:
            Dict with keys for all parameter sets
        """
        # Standard parameters (9, 26, 52)
        standard = self.ichimoku(high, low, close, 9, 26, 52, 26)
        
        # Crypto parameters (10, 30, 60) - REASON: crypto markets are 24/7
        # Used for CONF signals
        crypto = self.ichimoku(high, low, close, 10, 30, 60, 30)
        
        # Fast variant (7, 22, 44) - used for COIN signals
        fast_7 = self.ichimoku(high, low, close, 7, 22, 44, 22)
        
        # Alternative fast variant (6, 20, 52) - used for LEAD signals
        fast_6 = self.ichimoku(high, low, close, 6, 20, 52, 20)
        
        return {
            # Standard 9_26_52
            'ichi_tenkan_9_26_52': standard['tenkan_sen'],
            'ichi_kijun_9_26_52': standard['kijun_sen'],
            'ichi_senkou_a_9_26_52': standard['senkou_span_a'],
            'ichi_senkou_b_9_26_52': standard['senkou_span_b'],
            'ichi_chikou_9_26_52': standard['chikou_span'],
            # Crypto 10_30_60 (CONF signals)
            'ichi_tenkan_10_30_60': crypto['tenkan_sen'],
            'ichi_kijun_10_30_60': crypto['kijun_sen'],
            'ichi_senkou_a_10_30_60': crypto['senkou_span_a'],
            'ichi_senkou_b_10_30_60': crypto['senkou_span_b'],
            'ichi_chikou_10_30_60': crypto['chikou_span'],
            # Fast 7_22_44 (COIN signals)
            'ichi_tenkan_7_22_44': fast_7['tenkan_sen'],
            'ichi_kijun_7_22_44': fast_7['kijun_sen'],
            'ichi_senkou_a_7_22_44': fast_7['senkou_span_a'],
            'ichi_senkou_b_7_22_44': fast_7['senkou_span_b'],
            'ichi_chikou_7_22_44': fast_7['chikou_span'],
            # Alternative fast 6_20_52 (LEAD signals)
            'ichi_tenkan_6_20_52': fast_6['tenkan_sen'],
            'ichi_kijun_6_20_52': fast_6['kijun_sen'],
            'ichi_senkou_a_6_20_52': fast_6['senkou_span_a'],
            'ichi_senkou_b_6_20_52': fast_6['senkou_span_b'],
            'ichi_chikou_6_20_52': fast_6['chikou_span'],
        }
    
    def _calculate_all_smas_device(self, close_d: 'cp.ndarray | np.ndarray') -> Dict[str, np.ndarray]:
        """
        Calculate all SMA variants from device array.

        OPTIMIZED: Assumes close_d is already on device as float64.

        Args:
            close_d: Close prices already on device

        Returns:
            Dict with keys: sma_7, sma_25, sma_50, sma_99, sma_200
        """
        periods = {'sma_7': 7, 'sma_25': 25, 'sma_50': 50, 'sma_99': 99, 'sma_200': 200}
        return {
            name: self._to_host(self.sma(close_d, period, _device=True))
            for name, period in periods.items()
        }

    def _calculate_ichimoku_variants_device(
        self,
        high_d: 'cp.ndarray | np.ndarray',
        low_d: 'cp.ndarray | np.ndarray',
        close_d: 'cp.ndarray | np.ndarray'
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all Ichimoku variants from device arrays.

        OPTIMIZED: Assumes inputs are already on device as float64.

        Args:
            high_d, low_d, close_d: OHLC arrays already on device

        Returns:
            Dict with all Ichimoku variants (4 parameter sets × 5 components)
        """
        # Standard parameters (9, 26, 52)
        standard = self.ichimoku(high_d, low_d, close_d, 9, 26, 52, 26, _device=True)

        # Crypto parameters (10, 30, 60) - for CONF signals
        crypto = self.ichimoku(high_d, low_d, close_d, 10, 30, 60, 30, _device=True)

        # Fast variant (7, 22, 44) - for COIN signals
        fast_7 = self.ichimoku(high_d, low_d, close_d, 7, 22, 44, 22, _device=True)

        # Alternative fast variant (6, 20, 52) - for LEAD signals
        fast_6 = self.ichimoku(high_d, low_d, close_d, 6, 20, 52, 20, _device=True)

        return {
            # Standard 9_26_52
            'ichi_tenkan_9_26_52': standard['tenkan_sen'],
            'ichi_kijun_9_26_52': standard['kijun_sen'],
            'ichi_senkou_a_9_26_52': standard['senkou_span_a'],
            'ichi_senkou_b_9_26_52': standard['senkou_span_b'],
            'ichi_chikou_9_26_52': standard['chikou_span'],
            # Crypto 10_30_60 (CONF signals)
            'ichi_tenkan_10_30_60': crypto['tenkan_sen'],
            'ichi_kijun_10_30_60': crypto['kijun_sen'],
            'ichi_senkou_a_10_30_60': crypto['senkou_span_a'],
            'ichi_senkou_b_10_30_60': crypto['senkou_span_b'],
            'ichi_chikou_10_30_60': crypto['chikou_span'],
            # Fast 7_22_44 (COIN signals)
            'ichi_tenkan_7_22_44': fast_7['tenkan_sen'],
            'ichi_kijun_7_22_44': fast_7['kijun_sen'],
            'ichi_senkou_a_7_22_44': fast_7['senkou_span_a'],
            'ichi_senkou_b_7_22_44': fast_7['senkou_span_b'],
            'ichi_chikou_7_22_44': fast_7['chikou_span'],
            # Alternative fast 6_20_52 (LEAD signals)
            'ichi_tenkan_6_20_52': fast_6['tenkan_sen'],
            'ichi_kijun_6_20_52': fast_6['kijun_sen'],
            'ichi_senkou_a_6_20_52': fast_6['senkou_span_a'],
            'ichi_senkou_b_6_20_52': fast_6['senkou_span_b'],
            'ichi_chikou_6_20_52': fast_6['chikou_span'],
        }

    # =========================================================================
    # OBV - On Balance Volume
    # =========================================================================
    
    def obv(self, close: np.ndarray, volume: np.ndarray, *, _device: bool = False) -> np.ndarray:
        """
        Calculate On Balance Volume.

        OBV accumulates volume based on price direction.

        Args:
            close: Close prices
            volume: Volume
            _device: If True, assumes inputs are already on device as float64

        Returns:
            OBV values
        """
        xp = self._get_xp()
        if _device:
            close_d, volume_d = close, volume
        else:
            close_d = self._to_device_f64(close)
            volume_d = self._to_device_f64(volume)
        
        n = len(close_d)
        
        # Direction: +1 if close > prev_close, -1 if close < prev_close, 0 otherwise
        direction = xp.zeros(n, dtype=xp.float64)
        direction[1:] = xp.sign(close_d[1:] - close_d[:-1])
        
        # OBV = cumsum of direction * volume
        obv_values = xp.cumsum(direction * volume_d)
        
        return self._to_host(obv_values)
    
    # =========================================================================
    # CMF - Chaikin Money Flow
    # =========================================================================
    
    def cmf(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 20,
        *,
        _device: bool = False
    ) -> np.ndarray:
        """
        Calculate Chaikin Money Flow.

        CMF = Sum(MFV) / Sum(Volume) over period
        MFV = Money Flow Multiplier * Volume
        MFM = ((Close - Low) - (High - Close)) / (High - Low)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            period: CMF period (default 20)
            _device: If True, assumes inputs are already on device as float64

        Returns:
            CMF values (-1 to +1)
        """
        xp = self._get_xp()
        if _device:
            high_d, low_d, close_d, volume_d = high, low, close, volume
        else:
            high_d = self._to_device_f64(high)
            low_d = self._to_device_f64(low)
            close_d = self._to_device_f64(close)
            volume_d = self._to_device_f64(volume)
        
        n = len(close_d)
        cmf_values = xp.full(n, xp.nan, dtype=xp.float64)
        
        # Money Flow Multiplier
        hl_range = high_d - low_d
        mfm = xp.where(
            hl_range != 0,
            ((close_d - low_d) - (high_d - close_d)) / hl_range,
            0.0
        )
        
        # Money Flow Volume
        mfv = mfm * volume_d
        
        # Rolling sums
        mfv_cumsum = xp.cumsum(mfv)
        vol_cumsum = xp.cumsum(volume_d)
        
        if n >= period:
            mfv_sum = mfv_cumsum[period-1:] - xp.concatenate([xp.array([0.0]), mfv_cumsum[:-period]])
            vol_sum = vol_cumsum[period-1:] - xp.concatenate([xp.array([0.0]), vol_cumsum[:-period]])
            
            cmf_values[period-1:] = xp.where(vol_sum != 0, mfv_sum / vol_sum, 0.0)
        
        return self._to_host(cmf_values)
    
    # =========================================================================
    # Main calculation entry point
    # =========================================================================
    
    def calculate_all(
        self,
        ohlcv: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate all GPU-friendly indicators.

        OPTIMIZED: Pre-casts and pre-transfers all arrays to device once,
        then reuses them across all indicators.

        Args:
            ohlcv: Dict with keys: open, high, low, close, volume

        Returns:
            Dict with all calculated indicators
        """
        # ===== OPTIMIZATION: Cast and transfer to device ONCE =====
        high_d = self._to_device_f64(ohlcv['high'])
        low_d = self._to_device_f64(ohlcv['low'])
        close_d = self._to_device_f64(ohlcv['close'])
        volume_d = self._to_device_f64(ohlcv['volume'])

        results = {}

        # SMA variants - use device-aware helper
        results.update(self._calculate_all_smas_device(close_d))

        # Bollinger Bands - call with _device=True
        middle, upper, lower, bandwidth = self.bollinger_bands(
            close_d, period=20, std_dev=2.0, _device=True
        )
        results.update({
            'bb_middle': self._to_host(middle),
            'bb_upper': self._to_host(upper),
            'bb_lower': self._to_host(lower),
            'bb_bandwidth': self._to_host(bandwidth),
        })

        # Stochastic - call with _device=True
        stoch_k, stoch_d = self.stochastic(
            high_d, low_d, close_d, k_period=14, d_period=3, _device=True
        )
        results.update({
            'stoch_k': self._to_host(stoch_k),
            'stoch_d': self._to_host(stoch_d),
        })

        # Ichimoku (all variants: 9_26_52, 10_30_60, 7_22_44, 6_20_52) - use device-aware helper
        results.update(self._calculate_ichimoku_variants_device(high_d, low_d, close_d))

        # OBV - call with _device=True
        results['obv'] = self._to_host(self.obv(close_d, volume_d, _device=True))

        # CMF - call with _device=True
        results['cmf'] = self._to_host(
            self.cmf(high_d, low_d, close_d, volume_d, period=20, _device=True)
        )

        logger.debug(f"Calculated {len(results)} GPU indicators (optimized)")

        return results
