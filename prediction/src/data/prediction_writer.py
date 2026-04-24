"""
Prediction Writer for InfluxDB

Writes DAM and RTM predictions to InfluxDB for consumption by ercot-viewer.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from influxdb_client_3 import InfluxDBClient3, Point


class PredictionWriter:
    """Writes predictions to InfluxDB"""

    def __init__(
        self,
        url: str = None,
        token: str = None,
        org: str = None,
        database: str = None,
    ):
        """Initialize InfluxDB connection"""
        self.url = url or os.environ.get("INFLUXDB_URL")
        self.token = token or os.environ.get("INFLUXDB_TOKEN")
        self.org = org or os.environ.get("INFLUXDB_ORG")
        self.database = database or os.environ.get("INFLUXDB_BUCKET", "ercot")

        self.client = InfluxDBClient3(
            host=self.url.replace("https://", "").replace("http://", ""),
            token=self.token,
            org=self.org,
            database=self.database,
        )

    def close(self):
        """Close the client connection"""
        self.client.close()

    def write_dam_predictions(
        self,
        settlement_point: str,
        delivery_date: str,
        predictions: List[Dict[str, Any]],
        model_name: str = "simple_dam",
    ) -> int:
        """
        Write DAM price predictions to InfluxDB

        Args:
            settlement_point: Settlement point (e.g., "HB_HOUSTON")
            delivery_date: Delivery date in YYYY-MM-DD format
            predictions: List of predictions with hour_ending and predicted_price
            model_name: Name of the model used for predictions

        Returns:
            Number of points written
        """
        if not predictions:
            print("No predictions to write")
            return 0

        points = []
        generated_at = datetime.utcnow().isoformat() + "Z"

        for pred in predictions:
            try:
                hour_ending = pred.get("hour_ending")
                predicted_price = pred.get("predicted_price")

                if hour_ending is None or predicted_price is None:
                    continue

                # Parse hour from hour_ending (could be int or string like "01:00")
                if isinstance(hour_ending, str):
                    hour = int(hour_ending.split(":")[0])
                else:
                    hour = int(hour_ending)

                # Create timestamp in UTC (same convention as DAM actual prices)
                # delivery_date at hour-1 (hour ending convention) + 6 hours (CST to UTC)
                local_timestamp = datetime.fromisoformat(delivery_date)
                local_timestamp = local_timestamp.replace(hour=hour - 1)
                timestamp = local_timestamp + timedelta(hours=6)  # CST to UTC

                # Create point
                point = (
                    Point("dam_prediction")
                    .tag("settlement_point", settlement_point)
                    .tag("model", model_name)
                    .field("predicted_price", float(predicted_price))
                    .field("delivery_date", delivery_date)
                    .field("hour_ending", hour)
                    .field("generated_at", generated_at)
                    .time(timestamp)
                )

                points.append(point)

            except Exception as e:
                print(f"Error creating prediction point: {e}")
                continue

        if points:
            try:
                self.client.write(record=points)
                print(f"Wrote {len(points)} DAM predictions for {settlement_point} on {delivery_date}")
                return len(points)
            except Exception as e:
                print(f"Error writing predictions: {e}")
                return 0

        return 0

    def get_dam_predictions(
        self,
        settlement_point: str,
        delivery_date: str,
    ) -> List[Dict[str, Any]]:
        """
        Get DAM predictions from InfluxDB

        Args:
            settlement_point: Settlement point
            delivery_date: Delivery date in YYYY-MM-DD format

        Returns:
            List of predictions with hour_ending and predicted_price
        """
        # Calculate UTC time range for the delivery date
        # delivery_date hour 0 CST = hour 6 UTC
        # delivery_date hour 23 CST = delivery_date+1 hour 5 UTC
        date_obj = datetime.fromisoformat(delivery_date)
        start_utc = date_obj + timedelta(hours=6)  # 00:00 CST = 06:00 UTC
        end_utc = date_obj + timedelta(days=1, hours=5)  # 23:00 CST = 05:00 UTC next day

        query = f"""
        SELECT time, predicted_price, hour_ending
        FROM "dam_prediction"
        WHERE settlement_point = '{settlement_point}'
        AND time >= '{start_utc.isoformat()}Z'
        AND time <= '{end_utc.isoformat()}Z'
        ORDER BY time ASC
        """

        try:
            result = self.client.query(query)
            df = result.to_pandas()

            if df.empty:
                return []

            predictions = []
            for _, row in df.iterrows():
                hour = int(row['hour_ending'])
                predictions.append({
                    "hour_ending": f"{hour:02d}:00",
                    "predicted_price": float(row['predicted_price']),
                })

            return predictions

        except Exception as e:
            print(f"Error fetching predictions: {e}")
            return []


    def write_rtm_predictions(
        self,
        settlement_point: str,
        delivery_date: str,
        predictions: List[Dict[str, Any]],
        model_name: str = "rtm_1h",
    ) -> int:
        """
        Write RTM price predictions to InfluxDB.

        Same structure as write_dam_predictions but uses 'rtm_prediction' measurement.
        """
        if not predictions:
            return 0

        points = []
        generated_at = datetime.utcnow().isoformat() + "Z"

        for pred in predictions:
            try:
                hour_ending = pred.get("hour_ending")
                predicted_price = pred.get("predicted_price")

                if hour_ending is None or predicted_price is None:
                    continue

                if isinstance(hour_ending, str):
                    hour = int(hour_ending.split(":")[0])
                else:
                    hour = int(hour_ending)

                local_timestamp = datetime.fromisoformat(delivery_date)
                local_timestamp = local_timestamp.replace(hour=hour - 1)
                timestamp = local_timestamp + timedelta(hours=6)  # CST to UTC

                point = (
                    Point("rtm_prediction")
                    .tag("settlement_point", settlement_point)
                    .tag("model", model_name)
                    .field("predicted_price", float(predicted_price))
                    .field("delivery_date", delivery_date)
                    .field("hour_ending", hour)
                    .field("generated_at", generated_at)
                    .time(timestamp)
                )

                points.append(point)

            except Exception as e:
                print(f"Error creating RTM prediction point: {e}")
                continue

        if points:
            try:
                self.client.write(record=points)
                print(f"Wrote {len(points)} RTM predictions for {settlement_point} on {delivery_date}")
                return len(points)
            except Exception as e:
                print(f"Error writing RTM predictions: {e}")
                return 0

        return 0


def create_writer_from_env() -> PredictionWriter:
    """Create PredictionWriter from environment variables"""
    return PredictionWriter()
