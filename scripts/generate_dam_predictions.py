#!/usr/bin/env python3
"""
Generate DAM Predictions and Store in InfluxDB

This script generates DAM price predictions for the next day using the V2 predictor
and stores them in InfluxDB so they can be consumed by ercot-viewer.

Usage:
    python scripts/generate_dam_predictions.py [--date YYYY-MM-DD]
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.models.dam_v2_predictor import DAMV2Predictor
from src.data.influxdb_fetcher import create_fetcher_from_env
from src.data.prediction_writer import create_writer_from_env


# Settlement points to generate predictions for (matching ercot-viewer)
SETTLEMENT_POINTS = [
    "HB_BUSAVG", "HB_HOUSTON", "HB_HUBAVG", "HB_NORTH", "HB_PAN", "HB_SOUTH", "HB_WEST",
    "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST",
]


def get_tomorrow() -> str:
    """Get tomorrow's date in YYYY-MM-DD format"""
    tomorrow = datetime.now() + timedelta(days=1)
    return tomorrow.strftime("%Y-%m-%d")


def generate_predictions(settlement_point: str, target_date: str, predictor: DAMV2Predictor) -> list:
    """
    Generate DAM predictions for a settlement point

    Args:
        settlement_point: Settlement point code
        target_date: Target date in YYYY-MM-DD format
        predictor: V2 predictor instance

    Returns:
        List of predictions
    """
    if not predictor.is_ready():
        print(f"  V2 predictor not ready")
        return []

    # Fetch recent DAM data for features
    fetcher = create_fetcher_from_env()
    dam_df = fetcher.fetch_dam_prices(settlement_point=settlement_point)
    fetcher.close()

    if dam_df.empty:
        print(f"  No recent DAM data for {settlement_point}")
        return []

    print(f"  Loaded {len(dam_df)} records from InfluxDB")

    # Generate predictions
    target_dt = datetime.fromisoformat(target_date).date()
    predictions = predictor.predict_next_day(dam_df, target_dt)

    # Convert to dict format
    return [
        {
            "hour_ending": pred.hour_ending,
            "predicted_price": pred.predicted_price,
        }
        for pred in predictions
    ]


def main():
    parser = ArgumentParser(description="Generate DAM predictions using V2 models")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD). Default: tomorrow",
    )
    parser.add_argument(
        "--settlement-point",
        type=str,
        default=None,
        help="Single settlement point. Default: all configured points",
    )
    args = parser.parse_args()

    target_date = args.date or get_tomorrow()
    points = [args.settlement_point] if args.settlement_point else SETTLEMENT_POINTS

    print("=" * 60)
    print("DAM V2 Prediction Generator")
    print("=" * 60)
    print(f"Target date: {target_date}")
    print(f"Settlement points: {points}")
    print()

    # Load V2 predictor (24 per-hour models)
    print("Loading V2 predictor...")
    predictor = DAMV2Predictor()
    print(f"  Models loaded: {predictor.get_model_info()['models_loaded']}/24")
    print()

    writer = create_writer_from_env()
    total_written = 0

    for point in points:
        print(f"Processing {point}...")

        predictions = generate_predictions(point, target_date, predictor)

        if predictions:
            count = writer.write_dam_predictions(
                settlement_point=point,
                delivery_date=target_date,
                predictions=predictions,
                model_name="dam_v2",
            )
            total_written += count

            # Show sample predictions
            print(f"  Sample predictions:")
            for p in predictions[:3]:
                print(f"    Hour {p['hour_ending']:02d}: ${p['predicted_price']:.2f}")
            print(f"    ...")
        else:
            print(f"  No predictions generated")

        print()

    writer.close()
    print("=" * 60)
    print(f"Total predictions written: {total_written}")
    print("=" * 60)


if __name__ == "__main__":
    main()
