"""
VPVR (Volume Profile Visible Range) indicator calculations.

Calculates volume-at-price profile for each bar using a rolling lookback window.

Output columns for kfl.indicators:
- vpvr_poc: Point of Control (price level with highest volume)
- vpvr_vah: Value Area High (upper bound of 70% volume area)
- vpvr_val: Value Area Low (lower bound of 70% volume area)
- vpvr_hvn_upper: Upper High Volume Node
- vpvr_hvn_lower: Lower High Volume Node
- vpvr_lvn_upper: Upper Low Volume Node
- vpvr_lvn_lower: Lower Low Volume Node
"""

import logging
from typing import Dict, Tuple

import numpy as np
from numba import jit

logger = logging.getLogger(__name__)


# =============================================================================
# Numba JIT-compiled core functions
# =============================================================================

@jit(nopython=True, cache=True)
def _build_volume_histogram(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    start_idx: int,
    end_idx: int,
    num_bins: int
) -> Tuple[np.ndarray, float, float]:
    """
    Build volume histogram for a price range.
    
    Distributes each bar's volume across the price bins it spans.
    
    Args:
        high, low, close, volume: OHLCV arrays
        start_idx: Start index of window (inclusive)
        end_idx: End index of window (exclusive)
        num_bins: Number of price bins
        
    Returns:
        Tuple of (histogram, price_min, price_max)
    """
    # Find price range
    price_min = low[start_idx]
    price_max = high[start_idx]
    
    for i in range(start_idx, end_idx):
        if low[i] < price_min:
            price_min = low[i]
        if high[i] > price_max:
            price_max = high[i]
    
    # Handle edge case where price_min == price_max
    if price_max <= price_min:
        price_max = price_min + 0.0001
    
    bin_size = (price_max - price_min) / num_bins
    histogram = np.zeros(num_bins, dtype=np.float64)
    
    # Distribute volume across bins
    for i in range(start_idx, end_idx):
        bar_low = low[i]
        bar_high = high[i]
        bar_volume = volume[i]
        
        if bar_volume <= 0:
            continue
        
        # Find bins this bar spans
        low_bin = int((bar_low - price_min) / bin_size)
        high_bin = int((bar_high - price_min) / bin_size)
        
        # Clamp to valid range
        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))
        
        # Distribute volume evenly across bins
        num_spanned_bins = high_bin - low_bin + 1
        vol_per_bin = bar_volume / num_spanned_bins
        
        for b in range(low_bin, high_bin + 1):
            histogram[b] += vol_per_bin
    
    return histogram, price_min, price_max


@jit(nopython=True, cache=True)
def _build_volume_histogram_inplace(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    start_idx: int,
    end_idx: int,
    num_bins: int,
    histogram: np.ndarray  # Pre-allocated buffer
) -> Tuple[float, float, float]:
    """
    Build volume histogram IN-PLACE.

    OPTIMIZED: Reuses histogram buffer, returns total_vol to avoid recomputation.

    Args:
        high, low: Price arrays
        volume: Volume array
        start_idx: Start index of window (inclusive)
        end_idx: End index of window (exclusive)
        num_bins: Number of price bins
        histogram: Pre-allocated buffer to write histogram into

    Returns:
        Tuple of (price_min, bin_size, total_vol)
    """
    # Find price range
    price_min = low[start_idx]
    price_max = high[start_idx]
    for i in range(start_idx + 1, end_idx):
        if low[i] < price_min:
            price_min = low[i]
        if high[i] > price_max:
            price_max = high[i]

    if price_max <= price_min:
        price_max = price_min + 1e-8

    bin_size = (price_max - price_min) / num_bins

    # Zero histogram in-place
    for b in range(num_bins):
        histogram[b] = 0.0

    total_vol = 0.0

    # Distribute volume
    for i in range(start_idx, end_idx):
        bar_volume = volume[i]
        if bar_volume <= 0:
            continue

        total_vol += bar_volume

        # Find bins and distribute
        low_bin = int((low[i] - price_min) / bin_size)
        high_bin = int((high[i] - price_min) / bin_size)
        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))

        num_spanned = high_bin - low_bin + 1
        vol_per_bin = bar_volume / num_spanned

        for b in range(low_bin, high_bin + 1):
            histogram[b] += vol_per_bin

    return price_min, bin_size, total_vol


@jit(nopython=True, cache=True)
def _find_poc(histogram: np.ndarray, price_min: float, bin_size: float) -> float:
    """
    Find Point of Control (price level with highest volume).
    
    Returns the midpoint of the bin with maximum volume.
    """
    max_idx = 0
    max_vol = histogram[0]
    
    for i in range(1, len(histogram)):
        if histogram[i] > max_vol:
            max_vol = histogram[i]
            max_idx = i
    
    # Return midpoint of the bin
    return price_min + (max_idx + 0.5) * bin_size


@jit(nopython=True, cache=True)
def _find_value_area(
    histogram: np.ndarray,
    price_min: float,
    bin_size: float,
    value_area_pct: float,
    total_vol: float  # Pre-computed!
) -> Tuple[float, float]:
    """
    Find Value Area (price range containing specified % of volume).

    Expands outward from POC until value_area_pct of total volume is captured.

    OPTIMIZED: Accepts pre-computed total_vol to avoid recomputation.

    Returns:
        Tuple of (vah, val) - Value Area High and Low
    """
    if total_vol <= 0:
        mid_price = price_min + (len(histogram) / 2) * bin_size
        return mid_price, mid_price

    target_volume = total_vol * value_area_pct
    
    # Find POC bin
    poc_bin = 0
    max_vol = histogram[0]
    for i in range(1, len(histogram)):
        if histogram[i] > max_vol:
            max_vol = histogram[i]
            poc_bin = i
    
    # Expand from POC
    captured_volume = histogram[poc_bin]
    lower_bound = poc_bin
    upper_bound = poc_bin
    
    while captured_volume < target_volume:
        # Check which direction to expand
        can_expand_down = lower_bound > 0
        can_expand_up = upper_bound < len(histogram) - 1
        
        if not can_expand_down and not can_expand_up:
            break
        
        vol_below = histogram[lower_bound - 1] if can_expand_down else 0.0
        vol_above = histogram[upper_bound + 1] if can_expand_up else 0.0
        
        # Expand in direction with more volume
        if vol_below >= vol_above and can_expand_down:
            lower_bound -= 1
            captured_volume += vol_below
        elif can_expand_up:
            upper_bound += 1
            captured_volume += vol_above
        elif can_expand_down:
            lower_bound -= 1
            captured_volume += vol_below
        else:
            break
    
    val = price_min + lower_bound * bin_size
    vah = price_min + (upper_bound + 1) * bin_size
    
    return vah, val


@jit(nopython=True, cache=True)
def _find_hvn_lvn(
    histogram: np.ndarray,
    price_min: float,
    bin_size: float,
    mean_vol: float,  # Pre-computed!
    hvn_threshold: float,
    lvn_threshold: float
) -> Tuple[float, float, float, float]:
    """
    Find High Volume Nodes and Low Volume Nodes.

    HVN: Bins with volume > hvn_threshold * mean
    LVN: Bins with volume < lvn_threshold * mean

    OPTIMIZED: Accepts pre-computed mean_vol to avoid recomputation.

    Returns:
        Tuple of (hvn_upper, hvn_lower, lvn_upper, lvn_lower)
    """
    if mean_vol <= 0:
        mid_price = price_min + (len(histogram) / 2) * bin_size
        return mid_price, mid_price, mid_price, mid_price
    
    hvn_thresh = mean_vol * hvn_threshold
    lvn_thresh = mean_vol * lvn_threshold
    
    # Find HVN bins (high volume nodes)
    hvn_bins = []
    for i in range(len(histogram)):
        if histogram[i] >= hvn_thresh:
            hvn_bins.append(i)
    
    # Find LVN bins (low volume nodes)
    lvn_bins = []
    for i in range(len(histogram)):
        if histogram[i] <= lvn_thresh and histogram[i] > 0:
            lvn_bins.append(i)
    
    # Get upper and lower HVN
    if len(hvn_bins) > 0:
        hvn_lower = price_min + (hvn_bins[0] + 0.5) * bin_size
        hvn_upper = price_min + (hvn_bins[-1] + 0.5) * bin_size
    else:
        # Fallback to POC
        poc_bin = 0
        max_vol = histogram[0]
        for i in range(1, len(histogram)):
            if histogram[i] > max_vol:
                max_vol = histogram[i]
                poc_bin = i
        hvn_lower = hvn_upper = price_min + (poc_bin + 0.5) * bin_size
    
    # Get upper and lower LVN
    if len(lvn_bins) > 0:
        lvn_lower = price_min + (lvn_bins[0] + 0.5) * bin_size
        lvn_upper = price_min + (lvn_bins[-1] + 0.5) * bin_size
    else:
        # Fallback to edges
        lvn_lower = price_min + 0.5 * bin_size
        lvn_upper = price_min + (len(histogram) - 0.5) * bin_size
    
    return hvn_upper, hvn_lower, lvn_upper, lvn_lower


@jit(nopython=True, cache=True)
def _calculate_vpvr_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int,
    num_bins: int,
    value_area_pct: float,
    hvn_threshold: float,
    lvn_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core VPVR calculation for all bars.

    OPTIMIZED: Reuses histogram buffer and pre-computes total_vol/mean_vol.

    Returns arrays for: poc, vah, val, hvn_upper, hvn_lower, lvn_upper, lvn_lower
    """
    n = len(close)

    vpvr_poc = np.full(n, np.nan, dtype=np.float64)
    vpvr_vah = np.full(n, np.nan, dtype=np.float64)
    vpvr_val = np.full(n, np.nan, dtype=np.float64)
    vpvr_hvn_upper = np.full(n, np.nan, dtype=np.float64)
    vpvr_hvn_lower = np.full(n, np.nan, dtype=np.float64)
    vpvr_lvn_upper = np.full(n, np.nan, dtype=np.float64)
    vpvr_lvn_lower = np.full(n, np.nan, dtype=np.float64)

    # ALLOCATE HISTOGRAM ONCE
    histogram = np.empty(num_bins, dtype=np.float64)

    for i in range(lookback, n):
        start_idx = i - lookback
        end_idx = i

        # Build histogram in-place, get total_vol
        price_min, bin_size, total_vol = _build_volume_histogram_inplace(
            high, low, volume, start_idx, end_idx, num_bins, histogram
        )

        if total_vol <= 0:
            continue

        # Compute mean_vol once
        mean_vol = total_vol / num_bins

        # POC
        vpvr_poc[i] = _find_poc(histogram, price_min, bin_size)

        # Value Area - pass total_vol
        vah, val = _find_value_area(histogram, price_min, bin_size, value_area_pct, total_vol)
        vpvr_vah[i] = vah
        vpvr_val[i] = val

        # HVN/LVN - pass mean_vol
        hvn_upper, hvn_lower, lvn_upper, lvn_lower = _find_hvn_lvn(
            histogram, price_min, bin_size, mean_vol, hvn_threshold, lvn_threshold
        )
        vpvr_hvn_upper[i] = hvn_upper
        vpvr_hvn_lower[i] = hvn_lower
        vpvr_lvn_upper[i] = lvn_upper
        vpvr_lvn_lower[i] = lvn_lower

    return vpvr_poc, vpvr_vah, vpvr_val, vpvr_hvn_upper, vpvr_hvn_lower, vpvr_lvn_upper, vpvr_lvn_lower


# =============================================================================
# VPVRCalculator class
# =============================================================================

class VPVRCalculator:
    """
    Volume Profile Visible Range calculator.
    
    Calculates VPVR indicators using a rolling lookback window for each bar.
    Uses Numba JIT compilation for performance.
    
    Output columns (matching kfl.indicators schema):
    - vpvr_poc: Point of Control
    - vpvr_vah: Value Area High
    - vpvr_val: Value Area Low
    - vpvr_hvn_upper: Upper High Volume Node
    - vpvr_hvn_lower: Lower High Volume Node
    - vpvr_lvn_upper: Upper Low Volume Node
    - vpvr_lvn_lower: Lower Low Volume Node
    """
    
    def __init__(
        self,
        lookback: int = 50,
        num_bins: int = 50,
        value_area_pct: float = 0.70,
        hvn_threshold: float = 1.5,
        lvn_threshold: float = 0.5
    ):
        """
        Initialize VPVR calculator.
        
        Args:
            lookback: Number of bars for volume profile window
            num_bins: Number of price levels to divide range into
            value_area_pct: Percentage of volume for Value Area (default 70%)
            hvn_threshold: Multiplier above mean for High Volume Nodes
            lvn_threshold: Multiplier below mean for Low Volume Nodes
        """
        self.lookback = lookback
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.hvn_threshold = hvn_threshold
        self.lvn_threshold = lvn_threshold
        
        logger.info(
            f"VPVR calculator initialized: lookback={lookback}, bins={num_bins}, "
            f"VA={value_area_pct:.0%}, HVN>{hvn_threshold}x, LVN<{lvn_threshold}x"
        )
    
    def calculate_vpvr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate VPVR for each bar using rolling window.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            
        Returns:
            Dict with keys matching kfl.indicators columns:
            - vpvr_poc, vpvr_vah, vpvr_val
            - vpvr_hvn_upper, vpvr_hvn_lower
            - vpvr_lvn_upper, vpvr_lvn_lower
        """
        # Ensure float64 for Numba (no-copy if already float64)
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        
        # Run core calculation
        poc, vah, val, hvn_upper, hvn_lower, lvn_upper, lvn_lower = _calculate_vpvr_core(
            high, low, close, volume,
            self.lookback,
            self.num_bins,
            self.value_area_pct,
            self.hvn_threshold,
            self.lvn_threshold
        )
        
        logger.debug(f"Calculated VPVR for {len(close)} bars (lookback={self.lookback})")
        
        return {
            'vpvr_poc': poc,
            'vpvr_vah': vah,
            'vpvr_val': val,
            'vpvr_hvn_upper': hvn_upper,
            'vpvr_hvn_lower': hvn_lower,
            'vpvr_lvn_upper': lvn_upper,
            'vpvr_lvn_lower': lvn_lower,
        }
    
    def calculate_all(self, ohlcv: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate VPVR from OHLCV dict.
        
        Args:
            ohlcv: Dict with keys: open, high, low, close, volume
            
        Returns:
            Dict with VPVR indicators
        """
        return self.calculate_vpvr(
            ohlcv['high'],
            ohlcv['low'],
            ohlcv['close'],
            ohlcv['volume']
        )
