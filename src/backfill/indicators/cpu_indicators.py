"""
CPU-optimized indicator calculations using Numba.

Recursive/IIR indicators that require sequential computation:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- EMA (Exponential Moving Average)
- DEMA (Double Exponential Moving Average)
- ATR (Average True Range)
- ADX (Average Directional Index)
- Supertrend
- AO (Awesome Oscillator)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from numba import jit, prange

logger = logging.getLogger(__name__)


# =============================================================================
# Numba JIT-compiled core functions
# =============================================================================

@jit(nopython=True, cache=True)
def _ema_core(data: np.ndarray, period: int) -> np.ndarray:
    """
    Core EMA calculation using Wilder's smoothing.
    
    EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}
    alpha = 2 / (period + 1)
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Zoek eerste non-NaN index
    start_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start_idx = i
            break
    
    if start_idx == -1 or n < start_idx + period:
        return result
    
    alpha = 2.0 / (period + 1)
    
    # Initialize with SMA of first 'period' valid values
    sma = 0.0
    for i in range(start_idx, start_idx + period):
        sma += data[i]
    sma /= period
    result[start_idx + period - 1] = sma
    
    # EMA from period onwards
    for i in range(start_idx + period, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    
    return result


@jit(nopython=True, cache=True)
def _multi_ema_core(data: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    Calculate multiple EMAs in a single pass.

    OPTIMIZED: Shares data iteration across all periods.

    Args:
        data: Input array
        periods: Array of periods to calculate (e.g. [7, 25, 50, 99, 200])

    Returns:
        2D array of shape (len(data), len(periods))
    """
    n = len(data)
    num_periods = len(periods)
    results = np.empty((n, num_periods), dtype=np.float64)
    results[:, :] = np.nan

    # Pre-calculate alphas
    alphas = np.empty(num_periods, dtype=np.float64)
    for p_idx in range(num_periods):
        alphas[p_idx] = 2.0 / (periods[p_idx] + 1)

    # Initialize each EMA with SMA
    for p_idx in range(num_periods):
        period = periods[p_idx]
        if n < period:
            continue
        sma = 0.0
        for i in range(period):
            sma += data[i]
        sma /= period
        results[period - 1, p_idx] = sma

    # Calculate all EMAs in parallel in single loop
    for i in range(1, n):
        for p_idx in range(num_periods):
            if i >= periods[p_idx] and not np.isnan(results[i - 1, p_idx]):
                results[i, p_idx] = alphas[p_idx] * data[i] + (1 - alphas[p_idx]) * results[i - 1, p_idx]

    return results


@jit(nopython=True, cache=True)
def _rma_core(data: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's RMA (Running Moving Average) / SMMA.
    
    RMA_t = (RMA_{t-1} * (period-1) + value_t) / period
    Equivalent to EMA with alpha = 1/period
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    # Zoek eerste non-NaN index
    start_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start_idx = i
            break
    
    if start_idx == -1 or n < start_idx + period:
        return result
    
    alpha = 1.0 / period
    
    # Initialize with SMA
    sma = 0.0
    for i in range(start_idx, start_idx + period):
        sma += data[i]
    sma /= period
    result[start_idx + period - 1] = sma
    
    # RMA from period onwards
    for i in range(start_idx + period, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    
    return result


@jit(nopython=True, cache=True)
def _rsi_core(close: np.ndarray, period: int) -> np.ndarray:
    """
    RSI calculation using Wilder's smoothing.
    
    RS = avg_gain / avg_loss
    RSI = 100 - 100 / (1 + RS)
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n < period + 1:
        return result
    
    # Calculate price changes
    changes = np.empty(n, dtype=np.float64)
    changes[0] = 0.0
    for i in range(1, n):
        changes[i] = close[i] - close[i - 1]
    
    # Separate gains and losses
    gains = np.empty(n, dtype=np.float64)
    losses = np.empty(n, dtype=np.float64)
    for i in range(n):
        if changes[i] > 0:
            gains[i] = changes[i]
            losses[i] = 0.0
        else:
            gains[i] = 0.0
            losses[i] = -changes[i]
    
    # Initial averages (SMA of first period)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period
    
    # First RSI value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    
    # Subsequent RSI values using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    
    return result


@jit(nopython=True, cache=True)
def _multi_rsi_core(close: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    Calculate multiple RSIs in a single pass.

    OPTIMIZED: Shares changes/gains/losses computation across periods.

    Args:
        close: Close prices
        periods: Array of periods to calculate (e.g. [14, 7])

    Returns:
        2D array of shape (len(close), len(periods))
    """
    n = len(close)
    num_periods = len(periods)
    results = np.empty((n, num_periods), dtype=np.float64)
    results[:, :] = np.nan

    if n < 2:
        return results

    # Calculate changes ONCE (shared across all periods)
    changes = np.empty(n, dtype=np.float64)
    changes[0] = 0.0
    for i in range(1, n):
        changes[i] = close[i] - close[i - 1]

    # Separate gains and losses ONCE
    gains = np.empty(n, dtype=np.float64)
    losses = np.empty(n, dtype=np.float64)
    for i in range(n):
        if changes[i] > 0:
            gains[i] = changes[i]
            losses[i] = 0.0
        else:
            gains[i] = 0.0
            losses[i] = -changes[i]

    # Calculate RSI for each period using Wilder's smoothing
    for p_idx in range(num_periods):
        period = periods[p_idx]
        if n < period + 1:
            continue

        # Initial averages (SMA of first period)
        avg_gain = 0.0
        avg_loss = 0.0
        for i in range(1, period + 1):
            avg_gain += gains[i]
            avg_loss += losses[i]
        avg_gain /= period
        avg_loss /= period

        # First RSI value
        if avg_loss == 0:
            results[period, p_idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            results[period, p_idx] = 100.0 - 100.0 / (1.0 + rs)

        # Subsequent RSI values using Wilder's smoothing
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                results[i, p_idx] = 100.0
            else:
                rs = avg_gain / avg_loss
                results[i, p_idx] = 100.0 - 100.0 / (1.0 + rs)

    return results


@jit(nopython=True, cache=True)
def _macd_core(
    close: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD calculation.
    
    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD - Signal
    """
    n = len(close)
    macd_line = np.empty(n, dtype=np.float64)
    signal_line = np.empty(n, dtype=np.float64)
    histogram = np.empty(n, dtype=np.float64)
    macd_line[:] = np.nan
    signal_line[:] = np.nan
    histogram[:] = np.nan
    
    if n < slow_period:
        return macd_line, signal_line, histogram
    
    # Calculate EMAs
    fast_ema = _ema_core(close, fast_period)
    slow_ema = _ema_core(close, slow_period)
    
    # MACD Line
    for i in range(slow_period - 1, n):
        if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]):
            macd_line[i] = fast_ema[i] - slow_ema[i]
    
    # Signal Line (EMA of MACD)
    # Find first valid MACD value
    start_idx = slow_period - 1
    while start_idx < n and np.isnan(macd_line[start_idx]):
        start_idx += 1
    
    if start_idx + signal_period <= n:
        # Create temp array for signal EMA
        macd_valid = macd_line[start_idx:]
        signal_temp = _ema_core(macd_valid, signal_period)
        
        for i in range(len(signal_temp)):
            if not np.isnan(signal_temp[i]):
                signal_line[start_idx + i] = signal_temp[i]
    
    # Histogram
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram


@jit(nopython=True, cache=True)
def _dema_core(data: np.ndarray, period: int) -> np.ndarray:
    """
    Double EMA calculation.
    
    DEMA = 2 * EMA - EMA(EMA)
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan
    
    if n < period * 2:
        return result
    
    ema1 = _ema_core(data, period)
    ema2 = _ema_core(ema1, period)
    
    for i in range(n):
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i]):
            result[i] = 2.0 * ema1[i] - ema2[i]
    
    return result


@jit(nopython=True, cache=True)
def _multi_dema_core(data: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    Calculate multiple DEMAs in a single pass.

    OPTIMIZED: Calculates EMA and EMA(EMA) for all periods efficiently.

    DEMA = 2 * EMA - EMA(EMA)

    Args:
        data: Input array
        periods: Array of periods to calculate (e.g. [7, 25, 50, 99, 200])

    Returns:
        2D array of shape (len(data), len(periods))
    """
    n = len(data)
    num_periods = len(periods)
    results = np.empty((n, num_periods), dtype=np.float64)
    results[:, :] = np.nan

    # For each period: calculate EMA and EMA(EMA)
    for p_idx in range(num_periods):
        period = periods[p_idx]
        if n < period * 2:
            continue

        # First EMA
        ema1 = _ema_core(data, period)

        # Second EMA on first EMA
        ema2 = _ema_core(ema1, period)

        # DEMA
        for i in range(n):
            if not np.isnan(ema1[i]) and not np.isnan(ema2[i]):
                results[i, p_idx] = 2.0 * ema1[i] - ema2[i]

    return results


@jit(nopython=True, cache=True)
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Calculate True Range.
    
    TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    """
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    return tr


@jit(nopython=True, cache=True)
def _atr_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    ATR using Wilder's smoothing (RMA of True Range).
    """
    tr = _true_range(high, low, close)
    return _rma_core(tr, period)


@jit(nopython=True, cache=True)
def _adx_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ADX calculation.
    
    Returns: (ADX, +DI, -DI)
    """
    n = len(close)
    adx = np.empty(n, dtype=np.float64)
    plus_di = np.empty(n, dtype=np.float64)
    minus_di = np.empty(n, dtype=np.float64)
    adx[:] = np.nan
    plus_di[:] = np.nan
    minus_di[:] = np.nan
    
    if n < period * 2:
        return adx, plus_di, minus_di
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with RMA
    tr = _true_range(high, low, close)
    atr = _rma_core(tr, period)
    smooth_plus_dm = _rma_core(plus_dm, period)
    smooth_minus_dm = _rma_core(minus_dm, period)
    
    # +DI and -DI
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            if not np.isnan(smooth_plus_dm[i]):
                plus_di[i] = (smooth_plus_dm[i] / atr[i]) * 100.0
            if not np.isnan(smooth_minus_dm[i]):
                minus_di[i] = (smooth_minus_dm[i] / atr[i]) * 100.0
    
    # DX
    dx = np.empty(n, dtype=np.float64)
    dx[:] = np.nan
    for i in range(n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100.0
    
    # ADX = RMA of DX
    adx_temp = _rma_core(dx, period)
    for i in range(n):
        if not np.isnan(adx_temp[i]):
            adx[i] = adx_temp[i]
    
    return adx, plus_di, minus_di


@jit(nopython=True, cache=True)
def _supertrend_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    multiplier: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supertrend calculation.
    
    Returns: (supertrend, direction)
    direction: 1 = uptrend, -1 = downtrend
    """
    n = len(close)
    supertrend = np.empty(n, dtype=np.float64)
    direction = np.empty(n, dtype=np.float64)
    supertrend[:] = np.nan
    direction[:] = np.nan
    
    # Find first valid ATR
    start = 0
    while start < n and np.isnan(atr[start]):
        start += 1
    
    if start >= n:
        return supertrend, direction
    
    # Initialize
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Final bands
    final_upper = np.empty(n, dtype=np.float64)
    final_lower = np.empty(n, dtype=np.float64)
    final_upper[:] = np.nan
    final_lower[:] = np.nan
    
    final_upper[start] = upper_band[start]
    final_lower[start] = lower_band[start]
    supertrend[start] = final_upper[start]
    direction[start] = -1  # Start bearish
    
    for i in range(start + 1, n):
        if np.isnan(atr[i]):
            continue
            
        # Final upper band
        if upper_band[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]
        
        # Final lower band
        if lower_band[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]
        
        # Supertrend and direction
        if not np.isnan(supertrend[i - 1]):
            if supertrend[i - 1] == final_upper[i - 1]:
                if close[i] > final_upper[i]:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1
            else:
                if close[i] < final_lower[i]:
                    supertrend[i] = final_upper[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_lower[i]
                    direction[i] = 1
        else:
            supertrend[i] = final_upper[i]
            direction[i] = -1
    
    return supertrend, direction


@jit(nopython=True, cache=True)
def _ao_core(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Awesome Oscillator - OPTIMIZED to O(n).

    AO = SMA(5, median price) - SMA(34, median price)
    median price = (high + low) / 2

    Uses cumsum trick to avoid nested loops.
    """
    n = len(high)
    ao = np.empty(n, dtype=np.float64)
    ao[:] = np.nan

    if n < 34:
        return ao

    median = (high + low) / 2.0

    # Cumsum for both SMAs (O(n) instead of O(n*period))
    cumsum = np.empty(n + 1, dtype=np.float64)
    cumsum[0] = 0.0
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + median[i]

    # SMA 5 using cumsum
    sma5 = np.empty(n, dtype=np.float64)
    sma5[:] = np.nan
    for i in range(4, n):
        sma5[i] = (cumsum[i + 1] - cumsum[i - 4]) / 5.0

    # SMA 34 using cumsum
    sma34 = np.empty(n, dtype=np.float64)
    sma34[:] = np.nan
    for i in range(33, n):
        sma34[i] = (cumsum[i + 1] - cumsum[i - 33]) / 34.0

    # AO
    for i in range(33, n):
        if not np.isnan(sma5[i]) and not np.isnan(sma34[i]):
            ao[i] = sma5[i] - sma34[i]

    return ao


# =============================================================================
# CPUIndicatorCalculator class
# =============================================================================

class CPUIndicatorCalculator:
    """
    CPU-optimized indicator calculations using Numba.
    
    Handles recursive/IIR indicators that benefit from CPU optimization.
    """
    
    def __init__(self):
        """Initialize CPU calculator."""
        logger.info("CPU indicator calculator initialized with Numba")
    
    # =========================================================================
    # EMA variants
    # =========================================================================
    
    def ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        return _ema_core(np.asarray(data, dtype=np.float64), period)
    
    def calculate_all_emas(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate all EMA variants.

        OPTIMIZED: Single pass for all EMAs.
        """
        close64 = np.asarray(close, dtype=np.float64)
        # REASON: Sync met DB schema periodes
        periods = np.array([10, 12, 20, 26, 50, 100, 200], dtype=np.int64)

        # Single pass for all EMAs
        results_2d = _multi_ema_core(close64, periods)

        return {
            'ema_10': results_2d[:, 0],
            'ema_12': results_2d[:, 1],
            'ema_20': results_2d[:, 2],
            'ema_26': results_2d[:, 3],
            'ema_50': results_2d[:, 4],
            'ema_100': results_2d[:, 5],
            'ema_200': results_2d[:, 6],
        }
    
    # =========================================================================
    # DEMA variants
    # =========================================================================
    
    def dema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate DEMA."""
        return _dema_core(np.asarray(data, dtype=np.float64), period)
    
    def calculate_all_demas(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate all DEMA variants.

        OPTIMIZED: Efficient multi-period calculation.
        """
        close64 = np.asarray(close, dtype=np.float64)
        # REASON: Sync met DB schema periodes
        periods = np.array([10, 20, 50, 100, 200], dtype=np.int64)

        # Multi-period DEMA calculation
        results_2d = _multi_dema_core(close64, periods)

        return {
            'dema_10': results_2d[:, 0],
            'dema_20': results_2d[:, 1],
            'dema_50': results_2d[:, 2],
            'dema_100': results_2d[:, 3],
            'dema_200': results_2d[:, 4],
        }
    
    # =========================================================================
    # RSI
    # =========================================================================
    
    def rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        return _rsi_core(np.asarray(close, dtype=np.float64), period)
    
    def calculate_rsi_variants(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate RSI variants.

        OPTIMIZED: shared gains/losses computation across periods.
        """
        close64 = np.asarray(close, dtype=np.float64)
        # REASON: RSI 21 toegevoegd voor database conformiteit
        periods = np.array([14, 7, 21], dtype=np.int64)

        results_2d = _multi_rsi_core(close64, periods)

        return {
            'rsi_14': results_2d[:, 0],
            'rsi_7': results_2d[:, 1],
            'rsi_21': results_2d[:, 2],
        }
    
    # =========================================================================
    # MACD
    # =========================================================================
    
    def macd(
        self,
        close: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD."""
        return _macd_core(np.asarray(close, dtype=np.float64), fast, slow, signal)
    
    def calculate_macd_variants(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate all MACD variants.
        
        Variants:
        - 12_26_9: Standard MACD
        - 6_13_4: Fast MACD for LEAD signals
        - 8_24_9: Medium-fast MACD for LEAD signals
        - 5_35_5: Long-term MACD for CONF signals
        - 20_50_15: Very slow MACD for CONF signals
        """
        results = {}
        
        # Define all variants: (fast, slow, signal)
        variants = {
            '12_26_9': (12, 26, 9),   # Standard
            '6_13_4': (6, 13, 4),     # Fast - LEAD signals
            '8_24_9': (8, 24, 9),     # Medium-fast - LEAD signals
            '5_35_5': (5, 35, 5),     # Long-term - CONF signals
            '20_50_15': (20, 50, 15), # Very slow - CONF signals
        }
        
        for name, (fast, slow, signal) in variants.items():
            macd_line, signal_line, histogram = self.macd(close, fast, slow, signal)
            
            # Use consistent naming: macd_{variant} for line, macd_{variant}_signal, macd_{variant}_histogram
            results[f'macd_{name}'] = macd_line
            results[f'macd_{name}_signal'] = signal_line
            results[f'macd_{name}_histogram'] = histogram
        
        return results
    
    # =========================================================================
    # ATR / ADX
    # =========================================================================
    
    def atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate ATR."""
        return _atr_core(
            np.asarray(high, dtype=np.float64),
            np.asarray(low, dtype=np.float64),
            np.asarray(close, dtype=np.float64),
            period
        )
    
    def adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ADX. Returns (ADX, +DI, -DI)."""
        return _adx_core(
            np.asarray(high, dtype=np.float64),
            np.asarray(low, dtype=np.float64),
            np.asarray(close, dtype=np.float64),
            period
        )
    
    def calculate_atr_adx_variants(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate ATR and ADX variants."""
        atr_14 = self.atr(high, low, close, 14)
        atr_10 = self.atr(high, low, close, 10)  # REASON: voor Keltner Channels
        adx_14, plus_di_14, minus_di_14 = self.adx(high, low, close, 14)
        
        # REASON: atr_ma = SMA-20 van ATR-14 (voor ratio-gebaseerde atr_signal)
        atr_ma = self._sma(atr_14, 20)
        
        return {
            'atr_14': atr_14,
            'atr_10': atr_10,
            'atr_ma': atr_ma,
            'adx_14': adx_14,
            'plus_di_14': plus_di_14,
            'minus_di_14': minus_di_14,
        }

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Simple Moving Average over een berekende indicator array.
        Handles leading NaNs by starting calculation from first non-NaN value.
        
        REASON: Fixed off-by-one error in array concatenation.
        """
        n = len(data)
        result = np.full(n, np.nan, dtype=np.float64)
        if n < period:
            return result
        
        # Zoek eerste niet-NaN index
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return result
        
        first_valid = np.argmax(valid_mask)
        
        # Extract valid data slice
        valid_data = data[first_valid:]
        valid_n = len(valid_data)
        
        if valid_n < period:
            return result
        
        # Cumsum benadering over valid data
        cumsum = np.cumsum(valid_data)
        
        # SMA = (cumsum[i] - cumsum[i-period]) / period
        # Start bij index period-1 (relatief aan valid_data)
        sma_len = valid_n - period + 1
        sma_valid = np.empty(sma_len, dtype=np.float64)
        sma_valid[0] = cumsum[period - 1] / period
        if sma_len > 1:
            sma_valid[1:] = (cumsum[period:] - cumsum[:-period]) / period
        
        # Map terug naar result array
        out_start = first_valid + period - 1
        result[out_start : out_start + sma_len] = sma_valid
        
        return result
    
    # =========================================================================
    # Supertrend
    # =========================================================================
    
    def supertrend(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr_period: int = 10,
        multiplier: float = 3.0,
        *,
        atr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Supertrend. Returns (supertrend, direction).

        OPTIMIZED: Accepts pre-calculated ATR to avoid recomputation.

        Args:
            high, low, close: Price arrays
            atr_period: ATR period (default 10)
            multiplier: ATR multiplier (default 3.0)
            atr: Optional pre-calculated ATR array. If None, will calculate.

        Returns:
            Tuple of (supertrend, direction)
        """
        high64 = np.asarray(high, dtype=np.float64)
        low64 = np.asarray(low, dtype=np.float64)
        close64 = np.asarray(close, dtype=np.float64)

        if atr is None:
            atr = self.atr(high64, low64, close64, atr_period)

        return _supertrend_core(high64, low64, close64, atr, multiplier)
    
    def calculate_supertrend_variants(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Supertrend variants.

        OPTIMIZED: ATR calculated once and reused for all multipliers.
        """
        high64 = np.asarray(high, dtype=np.float64)
        low64 = np.asarray(low, dtype=np.float64)
        close64 = np.asarray(close, dtype=np.float64)

        # ===== OPTIMIZATION: Calculate ATR once =====
        atr10 = self.atr(high64, low64, close64, 10)

        # Reuse ATR for all multipliers
        st1, dir1 = self.supertrend(high64, low64, close64, 10, 1.0, atr=atr10)
        st2, dir2 = self.supertrend(high64, low64, close64, 10, 2.0, atr=atr10)
        st3, dir3 = self.supertrend(high64, low64, close64, 10, 3.0, atr=atr10)

        return {
            'supertrend_10_1': st1,
            'supertrend_dir_10_1': dir1,
            'supertrend_10_2': st2,
            'supertrend_dir_10_2': dir2,
            'supertrend_10_3': st3,
            'supertrend_dir_10_3': dir3,
        }
    
    # =========================================================================
    # Awesome Oscillator
    # =========================================================================
    
    def ao(self, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Calculate Awesome Oscillator."""
        return _ao_core(np.asarray(high, dtype=np.float64), np.asarray(low, dtype=np.float64))
    
    # =========================================================================
    # Keltner Channels
    # =========================================================================
    
    def keltner_channel(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Keltner Channel using exact EMA and Wilder's RMA for ATR.
        
        REASON: Synchronize with real-time pipeline (v3) which uses EMA-20 for middle line.
        """
        middle = self.ema(close, ema_period)
        atr = self.atr(high, low, close, atr_period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return middle, upper, lower

    def calculate_keltner_variants(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate Keltner Channel variants."""
        middle, upper, lower = self.keltner_channel(high, low, close, 20, 10, 2.0)
        return {
            'keltner_middle': middle,
            'keltner_upper': upper,
            'keltner_lower': lower,
        }

    # =========================================================================
    # Main calculation entry point
    # =========================================================================
    
    def calculate_all(self, ohlcv: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate all CPU-optimized indicators.
        
        Args:
            ohlcv: Dict with keys: open, high, low, close, volume
            
        Returns:
            Dict with all calculated indicators
        """
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        results = {}
        
        # EMA variants
        results.update(self.calculate_all_emas(close))
        
        # DEMA variants
        results.update(self.calculate_all_demas(close))
        
        # RSI variants
        results.update(self.calculate_rsi_variants(close))
        
        # MACD variants
        results.update(self.calculate_macd_variants(close))
        
        # ATR / ADX
        results.update(self.calculate_atr_adx_variants(high, low, close))
        
        # Supertrend
        results.update(self.calculate_supertrend_variants(high, low, close))
        
        # Awesome Oscillator
        results['ao'] = self.ao(high, low)
        
        # Keltner Channels
        results.update(self.calculate_keltner_variants(high, low, close))
        
        logger.debug(f"Calculated {len(results)} CPU indicators")
        
        return results
