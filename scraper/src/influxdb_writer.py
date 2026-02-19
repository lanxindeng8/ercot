"""
InfluxDB Writer

Handles writing ERCOT data to InfluxDB Cloud Serverless (3.x).
Uses influxdb3-python for proper compatibility with InfluxDB 3.0.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from influxdb_client_3 import InfluxDBClient3, Point

# Rate limiting settings for InfluxDB Cloud free tier
# Free tier has strict limits: ~5MB/5min write quota
BATCH_SIZE = 1000  # Write in smaller batches
BATCH_DELAY_SECONDS = 5  # Delay between batches
MAX_RETRIES = 5  # Max retries on rate limit
RETRY_DELAY_SECONDS = 120  # Base wait time on 429 error (will increase with backoff)


class InfluxDBWriter:
    """Writer for InfluxDB Cloud Serverless with rate limiting protection"""

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        database: str,
    ):
        """
        Initialize InfluxDB writer

        Args:
            url: InfluxDB URL (host)
            token: InfluxDB authentication token
            org: InfluxDB organization
            database: InfluxDB database (bucket) name
        """
        self.url = url
        self.token = token
        self.org = org
        self.database = database

        self.client = InfluxDBClient3(
            host=url.replace("https://", "").replace("http://", ""),
            token=token,
            org=org,
            database=database,
        )

    def close(self):
        """Close the InfluxDB client connection"""
        self.client.close()

    def write_rtm_lmp_data(self, records: List[Dict[str, Any]], measurement: str = "rtm_lmp_api") -> int:
        """
        Write Real-Time Market LMP data to InfluxDB (API data).

        Args:
            records: List of RTM LMP records from ERCOT API
            measurement: Target measurement name (default: rtm_lmp_api)

        Returns:
            Number of points written
        """
        if not records:
            print("No RTM LMP records to write")
            return 0

        # Debug: print first record's keys to see field names
        if records:
            print(f"DEBUG: First RTM LMP record keys: {list(records[0].keys())}")

        points = []
        skipped = 0

        for record in records:
            try:
                # Parse timestamp - try both possible field name formats
                timestamp_str = record.get("SCEDTimestamp") or record.get("scedTimestamp")
                if not timestamp_str:
                    skipped += 1
                    continue

                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

                # Floor timestamp to 5-minute interval for clean data
                timestamp = timestamp.replace(second=0, microsecond=0)
                minute = (timestamp.minute // 5) * 5
                timestamp = timestamp.replace(minute=minute)

                # Get field values - try both formats
                settlement_point = record.get("SettlementPoint") or record.get("settlementPoint") or ""
                lmp = record.get("LMP") or record.get("lmp") or 0
                energy = record.get("EnergyComponent") or record.get("energyComponent") or 0
                congestion = record.get("CongestionComponent") or record.get("congestionComponent") or 0
                loss = record.get("LossComponent") or record.get("lossComponent") or 0

                # Create point
                point = (
                    Point(measurement)
                    .tag("settlement_point", settlement_point)
                    .field("lmp", float(lmp or 0))
                    .field("energy_component", float(energy or 0))
                    .field("congestion_component", float(congestion or 0))
                    .field("loss_component", float(loss or 0))
                    .time(timestamp)
                )

                points.append(point)

            except Exception as e:
                print(f"Error creating point for RTM LMP record: {e}")
                continue

        print(f"RTM LMP: Created {len(points)} points, skipped {skipped} records")

        if points:
            return self._write_points_with_rate_limit(points, "RTM_LMP")

        return 0

    def write_rtm_lmp_realtime(self, records: List[Dict[str, Any]]) -> int:
        """
        Write Real-Time Market LMP data from CDR (real-time) to InfluxDB.

        Args:
            records: List of RTM LMP records from CDR scraper

        Returns:
            Number of points written
        """
        if not records:
            print("No RTM LMP realtime records to write")
            return 0

        points = []
        skipped = 0

        for record in records:
            try:
                # Parse timestamp
                timestamp_str = record.get("SCEDTimestamp") or record.get("scedTimestamp")
                if not timestamp_str:
                    skipped += 1
                    continue

                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

                # Floor timestamp to 5-minute interval
                timestamp = timestamp.replace(second=0, microsecond=0)
                minute = (timestamp.minute // 5) * 5
                timestamp = timestamp.replace(minute=minute)

                # Get field values
                settlement_point = record.get("SettlementPoint") or record.get("settlementPoint") or ""
                lmp = record.get("LMP") or record.get("lmp") or 0

                # Create point for rtm_lmp_realtime measurement
                point = (
                    Point("rtm_lmp_realtime")
                    .tag("settlement_point", settlement_point)
                    .field("lmp", float(lmp or 0))
                    .time(timestamp)
                )

                points.append(point)

            except Exception as e:
                print(f"Error creating point for RTM LMP realtime record: {e}")
                continue

        print(f"RTM LMP Realtime: Created {len(points)} points, skipped {skipped} records")

        if points:
            return self._write_points_with_rate_limit(points, "RTM_LMP_REALTIME")

        return 0

    def _write_points_with_rate_limit(self, points: List[Point], data_type: str) -> int:
        """Write points in batches with rate limiting protection and exponential backoff"""
        total_written = 0
        total_points = len(points)

        # Split into batches
        for i in range(0, total_points, BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_points + BATCH_SIZE - 1) // BATCH_SIZE

            # Retry logic with exponential backoff
            delay = RETRY_DELAY_SECONDS
            for retry in range(MAX_RETRIES):
                try:
                    self.client.write(record=batch)
                    total_written += len(batch)
                    print(f"Wrote batch {batch_num}/{total_batches}: {len(batch)} {data_type} points")
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "too many requests" in error_str.lower():
                        if retry < MAX_RETRIES - 1:
                            print(f"Rate limited, waiting {delay}s before retry {retry + 2}/{MAX_RETRIES}...")
                            time.sleep(delay)
                            delay = min(delay * 2, 600)  # Exponential backoff up to 10 min
                        else:
                            print(f"Max retries exceeded for batch {batch_num}. Wrote {total_written}/{total_points} points.")
                            return total_written
                    else:
                        print(f"Error writing {data_type} points: {e}")
                        raise

            # Delay between batches to avoid rate limiting
            if i + BATCH_SIZE < total_points:
                time.sleep(BATCH_DELAY_SECONDS)

        print(f"Successfully wrote {total_written} {data_type} points to InfluxDB")
        return total_written

    def write_dam_lmp_data(self, records: List[Dict[str, Any]]) -> int:
        """
        Write Day-Ahead Market LMP data to InfluxDB

        Args:
            records: List of DAM LMP records from ERCOT API

        Returns:
            Number of points written
        """
        if not records:
            print("No DAM LMP records to write")
            return 0

        # Debug: print first record's keys to see field names
        if records:
            print(f"DEBUG: First DAM LMP record keys: {list(records[0].keys())}")
            print(f"DEBUG: First DAM LMP record: {records[0]}")

        points = []
        skipped = 0

        for record in records:
            try:
                # Parse delivery date and hour - try both possible field name formats
                delivery_date = record.get("DeliveryDate") or record.get("deliveryDate")
                hour_ending = record.get("HourEnding") or record.get("hourEnding")

                if not delivery_date or not hour_ending:
                    skipped += 1
                    continue

                # Construct timestamp (delivery date + hour) in Central Time, then convert to UTC
                # HourEnding format: "01:00", "02:00", etc.
                # Hour ending 01:00 means the hour starting at 00:00 (midnight)
                hour = int(hour_ending.split(":")[0])

                # Create timestamp in Central Time (ERCOT operates in Central Time)
                # delivery_date is in format "YYYY-MM-DD"
                # hour - 1 because hour ending 01:00 means hour starting at 00:00
                local_timestamp = datetime.fromisoformat(delivery_date)
                local_timestamp = local_timestamp.replace(hour=hour - 1)

                # Convert Central Time to UTC (CST = UTC-6, CDT = UTC-5)
                # For simplicity, using CST (UTC-6). In production, use pytz for DST handling.
                from datetime import timedelta
                timestamp = local_timestamp + timedelta(hours=6)  # CST to UTC

                # Get settlement point fields - try both formats
                settlement_point = record.get("SettlementPoint") or record.get("settlementPoint") or ""
                settlement_point_type = record.get("SettlementPointType") or record.get("settlementPointType") or ""
                price = record.get("SettlementPointPrice") or record.get("settlementPointPrice") or 0

                # Create point
                point = (
                    Point("dam_lmp")
                    .tag("settlement_point", settlement_point)
                    .tag("settlement_point_type", settlement_point_type)
                    .field("lmp", float(price or 0))
                    .time(timestamp)
                )

                points.append(point)

            except Exception as e:
                print(f"Error creating point for DAM LMP record: {e}")
                continue

        print(f"DAM LMP: Created {len(points)} points, skipped {skipped} records")

        if points:
            return self._write_points_with_rate_limit(points, "DAM_LMP")

        return 0

    def get_last_timestamp(
        self, measurement: str, timestamp_field: str = "time"
    ) -> Optional[datetime]:
        """
        Get the last timestamp for a measurement using SQL

        Args:
            measurement: Measurement name (e.g., "rtm_lmp", "dam_lmp")
            timestamp_field: Timestamp field name

        Returns:
            Last timestamp or None if no data exists
        """
        query = f'''
        SELECT time FROM "{measurement}"
        ORDER BY time DESC
        LIMIT 1
        '''

        try:
            result = self.client.query(query)

            # Convert to pandas DataFrame and get the timestamp
            df = result.to_pandas()
            if not df.empty:
                last_time = df['time'].iloc[0]
                # Convert to datetime if needed
                if hasattr(last_time, 'to_pydatetime'):
                    return last_time.to_pydatetime()
                return last_time

            return None

        except Exception as e:
            print(f"Error querying last timestamp: {e}")
            return None


def create_writer_from_env() -> InfluxDBWriter:
    """
    Create InfluxDB writer from environment variables

    Required environment variables:
        - INFLUXDB_URL
        - INFLUXDB_TOKEN
        - INFLUXDB_ORG
        - INFLUXDB_BUCKET (or INFLUXDB_DATABASE)

    Returns:
        Configured InfluxDBWriter instance
    """
    return InfluxDBWriter(
        url=os.environ["INFLUXDB_URL"],
        token=os.environ["INFLUXDB_TOKEN"],
        org=os.environ["INFLUXDB_ORG"],
        database=os.environ.get("INFLUXDB_BUCKET") or os.environ.get("INFLUXDB_DATABASE", "ercot"),
    )
