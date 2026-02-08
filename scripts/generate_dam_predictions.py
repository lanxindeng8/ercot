#!/usr/bin/env python3
"""
Generate DAM Predictions and Store in InfluxDB

This script generates DAM price predictions for the next day and stores them
in InfluxDB so they can be consumed by ercot-viewer (hosted on Vercel).

Usage:
    python scripts/generate_dam_predictions.py [--date YYYY-MM-DD]
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from argparse import ArgumentParser

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from models.dam_simple_predictor import SimpleDAMPredictor
from data.influxdb_fetcher import create_fetcher_from_env
from data.prediction_writer import create_writer_from_env


# Settlement points to generate predictions for
SETTLEMENT_POINTS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"]


def get_tomorrow() -> str:
    """Get tomorrow's date in YYYY-MM-DD format"""
    tomorrow = datetime.now() + timedelta(days=1)
    return tomorrow.strftime("%Y-%m-%d")


def generate_predictions(settlement_point: str, target_date: str) -> list:
    """
    Generate DAM predictions for a settlement point

    Args:
        settlement_point: Settlement point code
        target_date: Target date in YYYY-MM-DD format

    Returns:
        List of predictions
    """
    model_dir = Path(__file__).parent.parent / "models"
    model_path = model_dir / f"dam_simple_{settlement_point.lower()}.joblib"

    if not model_path.exists():
        print(f"Model not found for {settlement_point}: {model_path}")
        return []

    # Load predictor
    predictor = SimpleDAMPredictor(model_path=model_path)

    if not predictor.is_ready():
        print(f"Model not ready for {settlement_point}")
        return []

    # Fetch recent DAM data for features
    fetcher = create_fetcher_from_env()
    dam_df = fetcher.fetch_dam_prices(
        settlement_point=settlement_point,
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
    )
    fetcher.close()

    if dam_df.empty:
        print(f"No recent DAM data for {settlement_point}")
        return []

    # Generate predictions
    target_dt = datetime.fromisoformat(target_date)
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
    parser = ArgumentParser(description="Generate DAM predictions")
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

    print(f"Generating DAM predictions for {target_date}")
    print(f"Settlement points: {points}")
    print()

    writer = create_writer_from_env()
    total_written = 0

    for point in points:
        print(f"Processing {point}...")

        predictions = generate_predictions(point, target_date)

        if predictions:
            count = writer.write_dam_predictions(
                settlement_point=point,
                delivery_date=target_date,
                predictions=predictions,
            )
            total_written += count
            print(f"  Wrote {count} predictions")
        else:
            print(f"  No predictions generated")

        print()

    writer.close()
    print(f"Total predictions written: {total_written}")


if __name__ == "__main__":
    main()
