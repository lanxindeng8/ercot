"""
Spike V2 Predictor — LightGBM zone-level spike detection.

Loads 14 per-settlement-point LightGBM models trained on lead_spike_60
and provides real-time spike probability predictions using the
spike_features feature builder.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import MODELS_DIR
from ..features.spike_features import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    SETTLEMENT_POINTS as SPIKE_SPS,
    build_spike_features,
)

log = logging.getLogger(__name__)

MODEL_DIR = MODELS_DIR / "spike"
DB_PATH = Path(__file__).resolve().parents[3] / "scraper" / "data" / "ercot_archive.db"

# Feature window: build 24h of features so rolling windows are warm
FEATURE_WINDOW_HOURS = 24

# Risk level thresholds
_RISK_LEVELS = [
    (0.8, "critical"),
    (0.6, "high"),
    (0.3, "medium"),
]


def _risk_level(prob: float) -> str:
    for threshold, level in _RISK_LEVELS:
        if prob >= threshold:
            return level
    return "low"


class SpikeV2Predictor:
    """Zone-level LightGBM spike predictor for all 14 settlement points."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.db_path = Path(db_path or DB_PATH)
        self.models: Dict[str, lgb.Booster] = {}
        self.metrics: Dict[str, Dict] = {}
        self._feature_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._load_models()

    def _load_models(self):
        """Load LightGBM models for all settlement points."""
        if not self.model_dir.exists():
            log.warning("Spike V2 model dir not found: %s", self.model_dir)
            return

        for sp in SPIKE_SPS:
            model_path = self.model_dir / f"{sp}_lead60.lgb"
            metrics_path = self.model_dir / f"{sp}_lead60_metrics.json"

            if not model_path.exists():
                log.warning("No spike V2 model for %s at %s", sp, model_path)
                continue

            try:
                model = lgb.Booster(model_file=str(model_path))
                self.models[sp] = model
                log.info("Loaded spike V2 model: %s", sp)
            except Exception as e:
                log.error("Failed to load spike V2 model %s: %s", sp, e)

            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        self.metrics[sp] = json.load(f)
                except Exception as e:
                    log.error("Failed to load metrics for %s: %s", sp, e)

        log.info(
            "Spike V2: loaded %d/%d models", len(self.models), len(SPIKE_SPS)
        )

    def is_ready(self) -> bool:
        return len(self.models) > 0

    def available_settlement_points(self) -> List[str]:
        return sorted(self.models.keys())

    def has_model(self, sp: str) -> bool:
        return sp.upper() in self.models

    def _get_features(self, sp: str) -> pd.DataFrame:
        """Build features for a settlement point, with caching."""
        now = time.time()
        if sp in self._feature_cache:
            cached_time, cached_df = self._feature_cache[sp]
            if now - cached_time < self._cache_ttl and not cached_df.empty:
                return cached_df

        end = datetime.now(timezone.utc)
        start = end - pd.Timedelta(hours=FEATURE_WINDOW_HOURS)

        try:
            df = build_spike_features(
                self.db_path,
                sp,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
            if not df.empty:
                self._feature_cache[sp] = (now, df)
            return df
        except Exception as e:
            log.error("Feature build failed for %s: %s", sp, e)
            # Return stale cache if available
            if sp in self._feature_cache:
                _, cached_df = self._feature_cache[sp]
                log.warning("Using stale feature cache for %s", sp)
                return cached_df
            return pd.DataFrame()

    def _top_drivers(self, sp: str, n: int = 3) -> List[str]:
        """Return top-N feature names by importance (gain) for this SP's model."""
        model = self.models.get(sp)
        if model is None:
            return []
        names = model.feature_name()
        importance = model.feature_importance(importance_type="gain")
        pairs = sorted(zip(names, importance), key=lambda x: x[1], reverse=True)
        return [name for name, _ in pairs[:n]]

    def predict(self, sp: str) -> Dict[str, Any]:
        """
        Predict spike probability for a single settlement point.

        Returns dict with probability, risk level, regime, top drivers, etc.
        """
        sp = sp.upper()
        if sp not in self.models:
            return {
                "settlement_point": sp,
                "error": f"No model loaded for {sp}",
                "probability": None,
                "is_alert": False,
                "risk_level": "unknown",
            }

        features_df = self._get_features(sp)
        if features_df.empty:
            return {
                "settlement_point": sp,
                "error": "No feature data available",
                "probability": None,
                "is_alert": False,
                "risk_level": "unknown",
            }

        # Get feature columns (exclude labels)
        feature_cols = [c for c in features_df.columns if c in FEATURE_COLUMNS]
        if not feature_cols:
            return {
                "settlement_point": sp,
                "error": "No valid feature columns",
                "probability": None,
                "is_alert": False,
                "risk_level": "unknown",
            }

        # Take the most recent row
        last_row = features_df[feature_cols].iloc[[-1]].fillna(0)

        model = self.models[sp]
        prob = float(model.predict(last_row)[0])
        prob = max(0.0, min(1.0, prob))  # clamp

        # Get regime from features if available
        regime = "Unknown"
        if "regime" in features_df.columns:
            regime = str(features_df["regime"].iloc[-1])

        risk = _risk_level(prob)
        now = datetime.now(timezone.utc)

        return {
            "settlement_point": sp,
            "probability": round(prob, 4),
            "is_alert": prob >= 0.3,
            "risk_level": risk,
            "regime": regime,
            "lead_time_minutes": 60,
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "model_version": "v2_lead60",
            "top_drivers": self._top_drivers(sp),
        }

    def predict_all(self) -> List[Dict[str, Any]]:
        """Predict for all loaded settlement points, sorted by probability desc."""
        results = []
        for sp in self.models:
            result = self.predict(sp)
            results.append(result)
        results.sort(
            key=lambda r: r.get("probability") or 0.0, reverse=True
        )
        return results

    def predict_alerts(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Return only predictions with probability >= threshold."""
        all_preds = self.predict_all()
        return [p for p in all_preds if (p.get("probability") or 0.0) >= threshold]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "Spike V2 LightGBM Binary Classifier",
            "target": "lead_spike_60",
            "models_loaded": len(self.models),
            "settlement_points": self.available_settlement_points(),
            "feature_count": len(FEATURE_COLUMNS),
            "lead_time_minutes": 60,
            "version": "v2_lead60",
        }


# Singleton
_spike_v2_predictor: Optional[SpikeV2Predictor] = None


def get_spike_v2_predictor() -> SpikeV2Predictor:
    global _spike_v2_predictor
    if _spike_v2_predictor is None:
        _spike_v2_predictor = SpikeV2Predictor()
    return _spike_v2_predictor
