"""
Gradient Boosting Model for Wind Forecasting

LightGBM-based quantile regression baseline.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseWindForecastModel

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None


class GBMWindModel(BaseWindForecastModel):
    """
    Gradient Boosting Model for wind generation forecasting.

    Uses LightGBM with quantile regression for uncertainty quantification.
    Trains separate models for each quantile.

    Advantages:
    - Fast training and inference
    - Handles missing values
    - Feature importance built-in
    - Good baseline performance
    """

    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        n_estimators: int = 1000,
        learning_rate: float = 0.02,
        max_depth: int = 8,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        use_gpu: bool = False,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
        verbose: int = -1,
    ):
        """
        Initialize GBM model.

        Args:
            quantiles: List of quantile levels to predict
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            num_leaves: Maximum number of leaves
            min_child_samples: Minimum samples per leaf
            subsample: Row subsampling ratio
            colsample_bytree: Column subsampling ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            use_gpu: Whether to use GPU acceleration
            random_state: Random seed
            early_stopping_rounds: Early stopping patience
            verbose: LightGBM verbosity (-1 for silent)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        self.quantiles = sorted(quantiles)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: Optional[List[str]] = None

    def _get_params(self, quantile: float) -> dict:
        """Get LightGBM parameters for a specific quantile."""
        params = {
            'objective': 'quantile',
            'alpha': quantile,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'n_jobs': -1,
        }

        if self.use_gpu:
            params['device'] = 'gpu'
            params['gpu_use_dp'] = False

        return params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> 'GBMWindModel':
        """
        Train separate LightGBM models for each quantile.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        for q in self.quantiles:
            logger.info(f"Training LightGBM for quantile {q}...")

            model = lgb.LGBMRegressor(**self._get_params(q))

            # Prepare callbacks
            callbacks = []
            if X_val is not None and y_val is not None:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
                callbacks.append(lgb.log_evaluation(period=100))

            # Train
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                callbacks=callbacks if callbacks else None,
            )

            self.models[q] = model
            logger.info(f"  Quantile {q}: {model.best_iteration_} iterations")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point forecast (median).

        Args:
            X: Features

        Returns:
            Point predictions (MW)
        """
        if 0.5 not in self.models:
            raise ValueError("Model not trained or median quantile (0.5) not available")

        return self.models[0.5].predict(X)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """
        Generate quantile predictions.

        Args:
            X: Features
            quantiles: Quantile levels (default: all trained quantiles)

        Returns:
            Dict mapping quantile to predictions
        """
        quantiles = quantiles or self.quantiles
        results = {}

        for q in quantiles:
            if q in self.models:
                results[q] = self.models[q].predict(X)
            else:
                logger.warning(f"Quantile {q} not available, skipping")

        return results

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance from median model.

        Returns:
            Series with feature importance
        """
        if 0.5 not in self.models or self.feature_names is None:
            return None

        importance = self.models[0.5].feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each quantile model
        for q, model in self.models.items():
            model_path = path / f"model_q{q:.2f}.txt"
            model.booster_.save_model(str(model_path))

        # Save metadata
        metadata = {
            'quantiles': self.quantiles,
            'feature_names': self.feature_names,
            'params': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'num_leaves': self.num_leaves,
            }
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'GBMWindModel':
        """
        Load model from disk.

        Args:
            path: Directory path

        Returns:
            Loaded model
        """
        path = Path(path)

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Create instance
        model = cls(
            quantiles=metadata['quantiles'],
            **metadata.get('params', {})
        )
        model.feature_names = metadata.get('feature_names')

        # Load quantile models
        for q in metadata['quantiles']:
            model_path = path / f"model_q{q:.2f}.txt"
            if model_path.exists():
                booster = lgb.Booster(model_file=str(model_path))
                # Wrap in LGBMRegressor-like interface
                lgb_model = lgb.LGBMRegressor()
                lgb_model._Booster = booster
                lgb_model._n_features = len(model.feature_names) if model.feature_names else 0
                model.models[q] = lgb_model

        logger.info(f"Model loaded from {path}")
        return model

    @property
    def name(self) -> str:
        return "LightGBM"
