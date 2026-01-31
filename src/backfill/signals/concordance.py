"""
Concordance calculation.

Aggregates signal strengths per classification to determine overall market sentiment.

Output format matches database schema:
- concordance_sum: Sum of all signal values
- concordance_count: Count of non-null signals
- concordance_score: Normalized score (-1 to +1)
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ConcordanceCalculator:
    """
    Calculates concordance scores from boolean and discrete signals.
    
    Concordance measures the degree of agreement among signals:
    - Positive concordance = majority bullish signals
    - Negative concordance = majority bearish signals
    - Low absolute concordance = mixed signals
    
    Output per table matches database schema:
    - concordance_sum: INTEGER - sum of signal values
    - concordance_count: INTEGER - number of non-null signals
    - concordance_score: NUMERIC - normalized score (-1.0 to +1.0)
    """
    
    def __init__(self):
        """Initialize concordance calculator."""
        logger.info("Concordance calculator initialized")
    
    def calculate_table_concordance(
        self,
        boolean_signals: Dict[str, np.ndarray],
        discrete_signals: Dict[str, np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate concordance metrics for a single table.
        
        Args:
            boolean_signals: Dict of signal_name -> signal_array (values: -1, 0, 1)
            discrete_signals: Optional dict of discrete signals (values: -2 to +2)
            
        Returns:
            Dict with:
            - concordance_sum: Sum of all signals (int)
            - concordance_count: Number of non-null signals (int)
            - concordance_score: Normalized score -1 to +1 (float)
        
        REASON: Optimized - max_possible is nu een scalar, geen array per signaal
        Dit vermijdt O(n * num_signals) array allocaties
        """
        if not boolean_signals and not discrete_signals:
            return {
                'concordance_sum': np.array([0], dtype=np.int32),
                'concordance_count': np.array([0], dtype=np.int32),
                'concordance_score': np.array([0.0], dtype=np.float64)
            }
        
        # Get array length from first available signal
        all_signals = {}
        if boolean_signals:
            all_signals.update(boolean_signals)
        if discrete_signals:
            all_signals.update(discrete_signals)
        
        n = len(next(iter(all_signals.values())))
        
        # REASON: int32 is voldoende: ~125 boolean + ~8 discrete = max ~141 per bar
        signal_sum = np.zeros(n, dtype=np.int32)
        signal_count = np.zeros(n, dtype=np.int32)
        
        # Process boolean signals (values -1, 0, +1)
        bool_count = 0
        if boolean_signals:
            bool_count = len(boolean_signals)
            for values in boolean_signals.values():
                # REASON: copy=False voorkomt onnodige kopie als dtype al compatible is
                signal_sum += values.astype(np.int32, copy=False)
                signal_count += (values != 0)  # bool -> int32 via ufunc
        
        # Process discrete signals (values -2 to +2)
        disc_count = 0
        if discrete_signals:
            disc_count = len(discrete_signals)
            for values in discrete_signals.values():
                signal_sum += values.astype(np.int32, copy=False)
                signal_count += (values != 0)
        
        # REASON: max_possible is een scalar, niet een array per signaal
        # boolean max = 1, discrete max = 2
        max_possible_scalar = bool_count + 2 * disc_count
        
        # Calculate normalized score
        if max_possible_scalar <= 0:
            concordance_score = np.zeros(n, dtype=np.float64)
        else:
            # REASON: float conversie alleen aan het eind
            concordance_score = np.clip(
                signal_sum.astype(np.float64) / float(max_possible_scalar),
                -1.0,
                1.0
            )
        
        return {
            'concordance_sum': signal_sum,
            'concordance_count': signal_count,
            'concordance_score': concordance_score
        }
    
    def calculate_all(
        self,
        boolean_signals: Dict[str, Dict[str, np.ndarray]],
        discrete_signals: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate concordance metrics for all tables.
        
        Args:
            boolean_signals: Dict with keys 'leading', 'coincident', 'confirming'
                Each value is a dict of signal_name -> signal_array
            discrete_signals: Dict with keys 'lead', 'coin', 'conf'
                Each value is a dict of discrete signal_name -> signal_array
            
        Returns:
            Dict with keys: 'lead', 'coin', 'conf'
            Each value contains:
            - concordance_sum
            - concordance_count
            - concordance_score
        """
        results = {}
        
        # LEAD table concordance
        lead_bool = boolean_signals.get('leading', {})
        lead_discrete = discrete_signals.get('lead', {})
        results['lead'] = self.calculate_table_concordance(lead_bool, lead_discrete)
        
        # COIN table concordance
        coin_bool = boolean_signals.get('coincident', {})
        coin_discrete = discrete_signals.get('coin', {})
        results['coin'] = self.calculate_table_concordance(coin_bool, coin_discrete)
        
        # CONF table concordance
        conf_bool = boolean_signals.get('confirming', {})
        conf_discrete = discrete_signals.get('conf', {})
        results['conf'] = self.calculate_table_concordance(conf_bool, conf_discrete)
        
        logger.debug(f"Calculated concordance for {len(results)} tables")
        
        return results
    
    def summarize_concordance(
        self,
        concordance_results: Dict[str, Dict[str, np.ndarray]],
        index: int = -1
    ) -> Dict[str, Dict[str, float]]:
        """
        Get concordance summary for a specific row (default: last row).
        
        Args:
            concordance_results: Output from calculate_all()
            index: Row index to summarize (default: -1 = last)
            
        Returns:
            Dict per table with scalar values for each metric
        """
        summary = {}
        
        for table, metrics in concordance_results.items():
            summary[table] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    summary[table][key] = float(value[index])
                else:
                    summary[table][key] = value
        
        return summary
