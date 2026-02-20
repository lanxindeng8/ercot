"""
Abstract Base Model Interface

All wind forecast models inherit from this interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BaseWindForecastModel(ABC):
    """
    Abstract base class for wind generation forecast models.

    All models must implement:
    1. fit() - Train on historical data
    2. predict() - Point forecast (p50/mean)
    3. predict_quantiles() - Uncertainty quantification
    4. save() / load() - Model persistence

    Optional methods:
    - predict_ramp() - Ramp risk assessment
    """

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> 'BaseWindForecastModel':
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets (wind generation MW)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point forecast (p50/mean).

        Args:
            X: Features for prediction

        Returns:
            Point forecasts (MW)
        """
        pass

    @abstractmethod
    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ) -> Dict[float, np.ndarray]:
        """
        Generate quantile forecasts.

        Args:
            X: Features for prediction
            quantiles: List of quantile levels (0-1)

        Returns:
            Dict mapping quantile level to predictions
        """
        pass

    def predict_interval(
        self,
        X: pd.DataFrame,
        coverage: float = 0.8,
    ) -> tuple:
        """
        Generate prediction interval.

        Args:
            X: Features for prediction
            coverage: Coverage probability (e.g., 0.8 for 80% interval)

        Returns:
            Tuple of (lower, upper) bounds
        """
        alpha = (1 - coverage) / 2
        quantiles = self.predict_quantiles(X, [alpha, 1 - alpha])
        return quantiles[alpha], quantiles[1 - alpha]

    def predict_ramp(
        self,
        X: pd.DataFrame,
        current_generation: float,
        threshold_mw: float = 2000,
    ) -> Dict[str, np.ndarray]:
        """
        Predict ramp magnitude and probability.

        Args:
            X: Features for prediction
            current_generation: Current wind generation (MW)
            threshold_mw: Threshold for significant ramp (MW)

        Returns:
            Dict with:
            - ramp_magnitude: Expected change (MW)
            - ramp_up_prob: P(ramp > threshold)
            - ramp_down_prob: P(ramp < -threshold)
        """
        # Point forecast
        forecast = self.predict(X)
        ramp_magnitude = forecast - current_generation

        # Use quantiles for probability estimation
        quantiles = self.predict_quantiles(X, [0.1, 0.9])
        upper_bound = quantiles[0.9]
        lower_bound = quantiles[0.1]

        # Estimate probabilities (simplified)
        # P(ramp_up) ~ proportion of uncertainty range above threshold
        ramp_up_prob = np.clip(
            (upper_bound - current_generation - threshold_mw) /
            (upper_bound - lower_bound + 1e-6),
            0, 1
        )

        # P(ramp_down) ~ proportion of uncertainty range below -threshold
        ramp_down_prob = np.clip(
            (current_generation - threshold_mw - lower_bound) /
            (upper_bound - lower_bound + 1e-6),
            0, 1
        )

        return {
            'ramp_magnitude': ramp_magnitude,
            'ramp_up_prob': ramp_up_prob,
            'ramp_down_prob': ramp_down_prob,
            'forecast_p50': forecast,
            'forecast_p10': lower_bound,
            'forecast_p90': upper_bound,
        }

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path for saving
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseWindForecastModel':
        """
        Load model from disk.

        Args:
            path: Directory path containing saved model

        Returns:
            Loaded model instance
        """
        pass

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores.

        Returns:
            Series with feature names as index and importance as values,
            or None if not supported by the model.
        """
        return None

    @property
    def name(self) -> str:
        """Model name for logging/display."""
        return self.__class__.__name__
