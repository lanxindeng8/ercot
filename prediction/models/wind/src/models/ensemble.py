"""
Ensemble Model for Wind Forecasting

Combines multiple models for improved predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseWindForecastModel


class EnsembleWindModel(BaseWindForecastModel):
    """
    Ensemble of wind forecasting models.

    Combines predictions from multiple models using weighted averaging.
    Supports both simple averaging and learned weights.
    """

    def __init__(
        self,
        models: Optional[List[BaseWindForecastModel]] = None,
        weights: Optional[List[float]] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9],
    ):
        """
        Initialize ensemble model.

        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
            quantiles: Quantile levels for predictions
        """
        self.models = models or []
        self.quantiles = sorted(quantiles)

        if weights is None and models:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights or []

        self._validate_weights()

    def _validate_weights(self):
        """Validate that weights sum to 1."""
        if self.weights and abs(sum(self.weights) - 1.0) > 1e-6:
            logger.warning("Weights don't sum to 1, normalizing...")
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def add_model(
        self,
        model: BaseWindForecastModel,
        weight: Optional[float] = None,
    ) -> 'EnsembleWindModel':
        """
        Add a model to the ensemble.

        Args:
            model: Trained model to add
            weight: Optional weight for this model

        Returns:
            self
        """
        self.models.append(model)

        if weight is not None:
            self.weights.append(weight)
            self._validate_weights()
        else:
            # Redistribute weights equally
            n = len(self.models)
            self.weights = [1.0 / n] * n

        return self

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> 'EnsembleWindModel':
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            self
        """
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.name}")
            model.fit(X_train, y_train, X_val, y_val)

        return self

    def fit_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        method: str = 'inverse_error',
    ) -> 'EnsembleWindModel':
        """
        Learn optimal weights from validation performance.

        Args:
            X_val: Validation features
            y_val: Validation targets
            method: Weight learning method ('inverse_error' or 'equal')

        Returns:
            self
        """
        if method == 'equal':
            self.weights = [1.0 / len(self.models)] * len(self.models)
            return self

        # Inverse error weighting
        errors = []
        for model in self.models:
            preds = model.predict(X_val)
            mae = np.mean(np.abs(y_val.values - preds))
            errors.append(mae)

        # Inverse of error (smaller error = higher weight)
        inv_errors = [1.0 / (e + 1e-8) for e in errors]
        total = sum(inv_errors)
        self.weights = [ie / total for ie in inv_errors]

        logger.info(f"Learned weights: {self.weights}")
        for i, (model, w, e) in enumerate(zip(self.models, self.weights, errors)):
            logger.info(f"  {model.name}: weight={w:.3f}, MAE={e:.1f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate weighted ensemble point forecast.

        Args:
            X: Features

        Returns:
            Ensemble predictions (MW)
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """
        Generate weighted ensemble quantile predictions.

        Args:
            X: Features
            quantiles: Quantile levels

        Returns:
            Dict mapping quantile to predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        quantiles = quantiles or self.quantiles
        results = {q: np.zeros(len(X)) for q in quantiles}

        for model, weight in zip(self.models, self.weights):
            model_preds = model.predict_quantiles(X, quantiles)
            for q in quantiles:
                if q in model_preds:
                    results[q] += model_preds[q] * weight

        return results

    def predict_all_models(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.

        Args:
            X: Features

        Returns:
            Dict mapping model name to predictions
        """
        results = {}
        for i, model in enumerate(self.models):
            key = f"{model.name}_{i}"
            results[key] = model.predict(X)
        return results

    def save(self, path: str) -> None:
        """
        Save ensemble to disk.

        Saves each model separately and a metadata file.

        Args:
            path: Directory path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for i, model in enumerate(self.models):
            model_path = path / f"model_{i}_{model.name}"
            model.save(str(model_path))

        # Save metadata
        import json
        metadata = {
            'n_models': len(self.models),
            'model_names': [m.name for m in self.models],
            'weights': self.weights,
            'quantiles': self.quantiles,
        }

        with open(path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'EnsembleWindModel':
        """
        Load ensemble from disk.

        Args:
            path: Directory path

        Returns:
            Loaded ensemble
        """
        import json
        from .gbm_model import GBMWindModel
        from .lstm_model import LSTMWindModel

        path = Path(path)

        # Load metadata
        with open(path / 'ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)

        # Model class mapping
        model_classes = {
            'LightGBM': GBMWindModel,
            'LSTM': LSTMWindModel,
        }

        # Load models
        models = []
        for i, name in enumerate(metadata['model_names']):
            model_path = path / f"model_{i}_{name}"

            if name in model_classes:
                model = model_classes[name].load(str(model_path))
                models.append(model)
            else:
                logger.warning(f"Unknown model type: {name}")

        ensemble = cls(
            models=models,
            weights=metadata['weights'],
            quantiles=metadata['quantiles'],
        )

        logger.info(f"Ensemble loaded from {path}")
        return ensemble

    @property
    def name(self) -> str:
        model_names = '+'.join(m.name for m in self.models)
        return f"Ensemble({model_names})"


class StackingEnsemble(BaseWindForecastModel):
    """
    Stacking ensemble that uses a meta-learner.

    First-level models make predictions, which are used as features
    for a second-level meta-learner.
    """

    def __init__(
        self,
        base_models: List[BaseWindForecastModel],
        meta_learner: Optional[BaseWindForecastModel] = None,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        include_original_features: bool = False,
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: First-level models
            meta_learner: Second-level model (default: GBM)
            quantiles: Quantile levels
            include_original_features: Include original features for meta-learner
        """
        self.base_models = base_models
        self.quantiles = sorted(quantiles)
        self.include_original_features = include_original_features

        if meta_learner is None:
            from .gbm_model import GBMWindModel
            self.meta_learner = GBMWindModel(quantiles=quantiles)
        else:
            self.meta_learner = meta_learner

    def _create_meta_features(
        self,
        X: pd.DataFrame,
        include_original: bool = False,
    ) -> pd.DataFrame:
        """
        Create meta-features from base model predictions.

        Args:
            X: Original features
            include_original: Whether to include original features

        Returns:
            Meta-features DataFrame
        """
        meta_features = {}

        for i, model in enumerate(self.base_models):
            # Point prediction
            meta_features[f'{model.name}_{i}_pred'] = model.predict(X)

            # Quantile predictions
            q_preds = model.predict_quantiles(X, self.quantiles)
            for q, pred in q_preds.items():
                meta_features[f'{model.name}_{i}_q{int(q*100)}'] = pred

        meta_df = pd.DataFrame(meta_features, index=X.index)

        if include_original:
            meta_df = pd.concat([X, meta_df], axis=1)

        return meta_df

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> 'StackingEnsemble':
        """
        Train stacking ensemble.

        Uses cross-validation to generate meta-features for training.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            self
        """
        # Train base models
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model.name}")
            model.fit(X_train, y_train, X_val, y_val)

        # Create meta-features
        logger.info("Creating meta-features...")
        meta_train = self._create_meta_features(X_train, self.include_original_features)

        meta_val = None
        if X_val is not None:
            meta_val = self._create_meta_features(X_val, self.include_original_features)

        # Train meta-learner
        logger.info(f"Training meta-learner: {self.meta_learner.name}")
        self.meta_learner.fit(meta_train, y_train, meta_val, y_val)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate stacked predictions.

        Args:
            X: Features

        Returns:
            Predictions (MW)
        """
        meta_features = self._create_meta_features(X, self.include_original_features)
        return self.meta_learner.predict(meta_features)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """
        Generate stacked quantile predictions.

        Args:
            X: Features
            quantiles: Quantile levels

        Returns:
            Dict mapping quantile to predictions
        """
        meta_features = self._create_meta_features(X, self.include_original_features)
        return self.meta_learner.predict_quantiles(meta_features, quantiles)

    def save(self, path: str) -> None:
        """Save stacking ensemble."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save base models
        for i, model in enumerate(self.base_models):
            model_path = path / f"base_{i}_{model.name}"
            model.save(str(model_path))

        # Save meta-learner
        meta_path = path / f"meta_{self.meta_learner.name}"
        self.meta_learner.save(str(meta_path))

        # Save metadata
        import json
        metadata = {
            'n_base_models': len(self.base_models),
            'base_model_names': [m.name for m in self.base_models],
            'meta_learner_name': self.meta_learner.name,
            'quantiles': self.quantiles,
            'include_original_features': self.include_original_features,
        }

        with open(path / 'stacking_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Stacking ensemble saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'StackingEnsemble':
        """Load stacking ensemble."""
        import json
        from .gbm_model import GBMWindModel
        from .lstm_model import LSTMWindModel

        path = Path(path)

        with open(path / 'stacking_metadata.json', 'r') as f:
            metadata = json.load(f)

        model_classes = {
            'LightGBM': GBMWindModel,
            'LSTM': LSTMWindModel,
        }

        # Load base models
        base_models = []
        for i, name in enumerate(metadata['base_model_names']):
            model_path = path / f"base_{i}_{name}"
            if name in model_classes:
                model = model_classes[name].load(str(model_path))
                base_models.append(model)

        # Load meta-learner
        meta_name = metadata['meta_learner_name']
        meta_path = path / f"meta_{meta_name}"
        if meta_name in model_classes:
            meta_learner = model_classes[meta_name].load(str(meta_path))
        else:
            meta_learner = GBMWindModel.load(str(meta_path))

        ensemble = cls(
            base_models=base_models,
            meta_learner=meta_learner,
            quantiles=metadata['quantiles'],
            include_original_features=metadata['include_original_features'],
        )

        logger.info(f"Stacking ensemble loaded from {path}")
        return ensemble

    @property
    def name(self) -> str:
        base_names = '+'.join(m.name for m in self.base_models)
        return f"Stacking({base_names}->{self.meta_learner.name})"
