"""
Configuration Management

YAML-based configuration for wind forecasting system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from loguru import logger


@dataclass
class DataConfig:
    """Data source configuration."""
    hrrr_source: str = 'aws'
    hrrr_cache: bool = True
    hrrr_max_workers: int = 12

    # Texas region bounds
    lat_min: float = 25.8
    lat_max: float = 36.5
    lon_min: float = -106.6
    lon_max: float = -93.5

    # ERCOT wind capacity (MW)
    wind_capacity: float = 40000.0


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Wind features
    wind_variables: List[str] = field(default_factory=lambda: [
        'u10m', 'v10m', 'u80m', 'v80m', 't2m', 'sp'
    ])

    # Power curve parameters
    cut_in_speed: float = 3.0
    rated_speed: float = 12.0
    cut_out_speed: float = 25.0

    # Temporal lags (hours)
    lag_hours: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])

    # Rolling windows (hours)
    rolling_windows: List[int] = field(default_factory=lambda: [6, 12, 24])


@dataclass
class ModelConfig:
    """Model training configuration."""
    # LightGBM
    gbm_n_estimators: int = 1000
    gbm_learning_rate: float = 0.02
    gbm_max_depth: int = 8
    gbm_num_leaves: int = 63
    gbm_early_stopping_rounds: int = 50

    # LSTM
    lstm_seq_length: int = 24
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 1e-3
    lstm_batch_size: int = 64
    lstm_epochs: int = 100
    lstm_patience: int = 10

    # Common
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    random_state: int = 42


@dataclass
class RampConfig:
    """Ramp detection configuration."""
    # Ramp thresholds (MW)
    ramp_threshold_small: float = 1000.0
    ramp_threshold_medium: float = 2000.0
    ramp_threshold_large: float = 3000.0

    # Detection windows (hours)
    ramp_window: int = 3
    ramp_tolerance: int = 2

    # No-solar period definition
    # Based on Texas typical sunset/sunrise
    no_solar_start_hour: int = 19  # 7 PM
    no_solar_end_hour: int = 7     # 7 AM

    # Evening peak hours
    evening_peak_start: int = 17   # 5 PM
    evening_peak_end: int = 21     # 9 PM


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data split
    train_start: str = '2023-01-01'
    train_end: str = '2023-12-31'
    val_start: str = '2024-01-01'
    val_end: str = '2024-03-31'
    test_start: str = '2024-04-01'
    test_end: str = '2024-06-30'

    # Forecast horizons (hours)
    horizons: List[int] = field(default_factory=lambda: list(range(1, 13)))

    # Output
    output_dir: str = 'outputs'
    model_dir: str = 'models'


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ramp: RampConfig = field(default_factory=RampConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Config object
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"Config file not found: {path}, using defaults")
        return Config()

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}

    # Build config from dict
    config = Config(
        data=DataConfig(**config_dict.get('data', {})),
        features=FeatureConfig(**config_dict.get('features', {})),
        model=ModelConfig(**config_dict.get('model', {})),
        ramp=RampConfig(**config_dict.get('ramp', {})),
        training=TrainingConfig(**config_dict.get('training', {})),
    )

    logger.info(f"Loaded config from {path}")
    return config


def save_config(config: Config, path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        'data': {
            'hrrr_source': config.data.hrrr_source,
            'hrrr_cache': config.data.hrrr_cache,
            'hrrr_max_workers': config.data.hrrr_max_workers,
            'lat_min': config.data.lat_min,
            'lat_max': config.data.lat_max,
            'lon_min': config.data.lon_min,
            'lon_max': config.data.lon_max,
            'wind_capacity': config.data.wind_capacity,
        },
        'features': {
            'wind_variables': config.features.wind_variables,
            'cut_in_speed': config.features.cut_in_speed,
            'rated_speed': config.features.rated_speed,
            'cut_out_speed': config.features.cut_out_speed,
            'lag_hours': config.features.lag_hours,
            'rolling_windows': config.features.rolling_windows,
        },
        'model': {
            'gbm_n_estimators': config.model.gbm_n_estimators,
            'gbm_learning_rate': config.model.gbm_learning_rate,
            'gbm_max_depth': config.model.gbm_max_depth,
            'gbm_num_leaves': config.model.gbm_num_leaves,
            'gbm_early_stopping_rounds': config.model.gbm_early_stopping_rounds,
            'lstm_seq_length': config.model.lstm_seq_length,
            'lstm_hidden_dim': config.model.lstm_hidden_dim,
            'lstm_num_layers': config.model.lstm_num_layers,
            'lstm_dropout': config.model.lstm_dropout,
            'lstm_learning_rate': config.model.lstm_learning_rate,
            'lstm_batch_size': config.model.lstm_batch_size,
            'lstm_epochs': config.model.lstm_epochs,
            'lstm_patience': config.model.lstm_patience,
            'quantiles': config.model.quantiles,
            'random_state': config.model.random_state,
        },
        'ramp': {
            'ramp_threshold_small': config.ramp.ramp_threshold_small,
            'ramp_threshold_medium': config.ramp.ramp_threshold_medium,
            'ramp_threshold_large': config.ramp.ramp_threshold_large,
            'ramp_window': config.ramp.ramp_window,
            'ramp_tolerance': config.ramp.ramp_tolerance,
            'no_solar_start_hour': config.ramp.no_solar_start_hour,
            'no_solar_end_hour': config.ramp.no_solar_end_hour,
            'evening_peak_start': config.ramp.evening_peak_start,
            'evening_peak_end': config.ramp.evening_peak_end,
        },
        'training': {
            'train_start': config.training.train_start,
            'train_end': config.training.train_end,
            'val_start': config.training.val_start,
            'val_end': config.training.val_end,
            'test_start': config.training.test_start,
            'test_end': config.training.test_end,
            'horizons': config.training.horizons,
            'output_dir': config.training.output_dir,
            'model_dir': config.training.model_dir,
        },
    }

    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {path}")


def create_default_config(path: str) -> Config:
    """
    Create and save default configuration.

    Args:
        path: Output path

    Returns:
        Default config
    """
    config = Config()
    save_config(config, path)
    return config
