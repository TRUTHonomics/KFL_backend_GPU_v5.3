"""
Signal calculation modules.

- discrete_signals: 8 graduele signals (-2 to +2)
- boolean_signals: 125 boolean signals per classificatie (LEAD/COIN/CONF)
- concordance: Concordance berekening per classificatie
"""

from .discrete_signals import DiscreteSignalCalculator
from .boolean_signals import BooleanSignalCalculator
from .concordance import ConcordanceCalculator

__all__ = ['DiscreteSignalCalculator', 'BooleanSignalCalculator', 'ConcordanceCalculator']
