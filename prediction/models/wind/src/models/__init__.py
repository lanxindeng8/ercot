"""
Models Module for Wind Forecasting

Collection of forecasting models for wind generation prediction.
"""

from .base import BaseWindForecastModel
from .gbm_model import GBMWindModel
from .lstm_model import LSTMWindModel, TORCH_AVAILABLE
from .ensemble import EnsembleWindModel, StackingEnsemble

__all__ = [
    'BaseWindForecastModel',
    'GBMWindModel',
    'LSTMWindModel',
    'EnsembleWindModel',
    'StackingEnsemble',
    'TORCH_AVAILABLE',
]
