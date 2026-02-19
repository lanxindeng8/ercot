"""
InfluxDB Data Fetcher

Fetches historical DAM and RTM price data from InfluxDB for model training and inference.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from influxdb_client_3 import InfluxDBClient3


class InfluxDBFetcher:
    """Fetches ERCOT price data from InfluxDB"""

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

    def fetch_dam_prices(
        self,
        settlement_point: str = "LZ_HOUSTON",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch DAM LMP prices from InfluxDB

        Args:
            settlement_point: ERCOT settlement point (e.g., LZ_HOUSTON)
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with columns: timestamp, hour, dam_price
        """
        if start_date is None:
            start_date = datetime(2015, 1, 1)
        if end_date is None:
            end_date = datetime.utcnow()

        query = f"""
        SELECT time, lmp
        FROM "dam_lmp"
        WHERE settlement_point = '{settlement_point}'
        AND time >= '{start_date.isoformat()}Z'
        AND time <= '{end_date.isoformat()}Z'
        ORDER BY time ASC
        """

        try:
            result = self.client.query(query)
            df = result.to_pandas()

            if df.empty:
                print(f"No DAM data found for {settlement_point}")
                return pd.DataFrame(columns=['timestamp', 'hour', 'dam_price'])

            # Process the data
            df = df.rename(columns={'time': 'timestamp', 'lmp': 'dam_price'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour + 1  # Convert to HE1-HE24
            df['date'] = df['timestamp'].dt.date
            df = df.set_index('timestamp')

            return df

        except Exception as e:
            print(f"Error fetching DAM prices: {e}")
            return pd.DataFrame(columns=['timestamp', 'hour', 'dam_price'])

    def fetch_rtm_prices(
        self,
        settlement_point: str = "LZ_HOUSTON",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch RTM LMP prices from InfluxDB

        Args:
            settlement_point: ERCOT settlement point
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            DataFrame with columns: timestamp, lmp, energy, congestion, loss
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()

        query = f"""
        SELECT time, lmp, energy_component, congestion_component, loss_component
        FROM "rtm_lmp"
        WHERE settlement_point = '{settlement_point}'
        AND time >= '{start_date.isoformat()}Z'
        AND time <= '{end_date.isoformat()}Z'
        ORDER BY time ASC
        """

        try:
            result = self.client.query(query)
            df = result.to_pandas()

            if df.empty:
                print(f"No RTM data found for {settlement_point}")
                return pd.DataFrame()

            df = df.rename(columns={
                'time': 'timestamp',
                'lmp': 'rtm_price',
                'energy_component': 'energy',
                'congestion_component': 'congestion',
                'loss_component': 'loss'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            return df

        except Exception as e:
            print(f"Error fetching RTM prices: {e}")
            return pd.DataFrame()

    def get_available_settlement_points(self, measurement: str = "dam_lmp") -> List[str]:
        """Get list of available settlement points"""
        query = f"""
        SELECT DISTINCT settlement_point
        FROM "{measurement}"
        """

        try:
            result = self.client.query(query)
            df = result.to_pandas()
            return df['settlement_point'].tolist() if not df.empty else []
        except Exception as e:
            print(f"Error getting settlement points: {e}")
            return []

    def get_data_range(self, measurement: str = "dam_lmp") -> Tuple[datetime, datetime]:
        """Get the time range of available data"""
        query = f"""
        SELECT MIN(time) as min_time, MAX(time) as max_time
        FROM "{measurement}"
        """

        try:
            result = self.client.query(query)
            df = result.to_pandas()
            if not df.empty:
                return (
                    pd.to_datetime(df['min_time'].iloc[0]),
                    pd.to_datetime(df['max_time'].iloc[0])
                )
        except Exception as e:
            print(f"Error getting data range: {e}")

        return None, None


def create_fetcher_from_env() -> InfluxDBFetcher:
    """Create InfluxDB fetcher from environment variables"""
    return InfluxDBFetcher()
