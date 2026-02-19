"""
ERCOT API Client

Handles authentication and data fetching from ERCOT APIs.
Based on the original Node.js Lambda implementation.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Generator, Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ErcotClient:
    """Client for interacting with ERCOT APIs"""

    # API Constants
    TOKEN_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
    PUBLIC_BASE_URL = "https://api.ercot.com/api/public-reports"
    ESR_BASE_URL = "https://api.ercot.com/api/esl-reports"

    # Endpoints
    ENDPOINT_LMP_SETTLEMENT_POINT = "/np6-788-cd/lmp_node_zone_hub"
    ENDPOINT_SPP_DAY_AHEAD = "/np4-190-cd/dam_stlmnt_pnt_prices"

    # Configuration
    DEFAULT_PAGE_SIZE = 50000
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_INITIAL_DELAY_MS = 1000
    TOKEN_EXPIRATION_SECONDS = 3600
    ERCOT_TIMEZONE = "America/Chicago"

    def __init__(
        self,
        username: str,
        password: str,
        public_subscription_key: str,
        esr_subscription_key: Optional[str] = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize ERCOT API client

        Args:
            username: ERCOT API username
            password: ERCOT API password
            public_subscription_key: Public API subscription key
            esr_subscription_key: ESR API subscription key (optional)
            page_size: Number of records per page
            max_retries: Maximum number of retries for failed requests
        """
        self.username = username
        self.password = password
        self.public_subscription_key = public_subscription_key
        self.esr_subscription_key = esr_subscription_key or public_subscription_key
        self.page_size = page_size
        self.max_retries = max_retries

        self.token: Optional[str] = None
        self.token_expiry: Optional[float] = None

        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_token(self) -> str:
        """
        Authenticate with ERCOT B2C and get access token

        Returns:
            Access token string
        """
        print("Requesting new ERCOT API token...")

        payload = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "response_type": "id_token",
            "scope": f"openid {self.CLIENT_ID} offline_access",
            "client_id": self.CLIENT_ID,
        }

        try:
            response = self.session.post(
                self.TOKEN_URL,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            if "id_token" not in data:
                raise ValueError("No id_token in response")

            self.token = data["id_token"]
            self.token_expiry = time.time() + self.TOKEN_EXPIRATION_SECONDS

            print("Successfully obtained ERCOT API token")
            return self.token

        except Exception as e:
            raise RuntimeError(f"Failed to obtain ERCOT API token: {str(e)}")

    def refresh_token_if_needed(self):
        """Refresh token if expired or not yet obtained"""
        if not self.token or not self.token_expiry or time.time() >= self.token_expiry:
            self.get_token()

    def get_headers(self, api_type: str = "public") -> Dict[str, str]:
        """
        Get headers for API requests

        Args:
            api_type: "public" or "esr"

        Returns:
            Dictionary of HTTP headers
        """
        self.refresh_token_if_needed()

        subscription_key = (
            self.public_subscription_key
            if api_type == "public"
            else self.esr_subscription_key
        )

        return {
            "Authorization": f"Bearer {self.token}",
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-Type": "application/json",
        }

    def make_api_call(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        api_type: str = "public",
    ) -> Dict[str, Any]:
        """
        Make an API call with automatic retry on rate limiting

        Args:
            url: Full API URL
            params: Query parameters
            api_type: "public" or "esr"

        Returns:
            API response data
        """
        headers = self.get_headers(api_type)

        retries = 0
        delay = self.DEFAULT_INITIAL_DELAY_MS / 1000

        while retries <= self.max_retries:
            try:
                response = self.session.get(
                    url, headers=headers, params=params, timeout=60
                )

                if response.status_code == 429:
                    # Rate limited
                    retries += 1
                    if retries <= self.max_retries:
                        print(f"Rate limited, retrying in {delay}s... (attempt {retries}/{self.max_retries})")
                        time.sleep(delay)
                        delay *= 2
                        continue
                    else:
                        raise RuntimeError("Max retries exceeded due to rate limiting")

                elif response.status_code == 401:
                    # Token expired
                    print("Token expired, refreshing...")
                    self.token = None
                    self.token_expiry = None
                    retries += 1
                    if retries <= self.max_retries:
                        continue
                    else:
                        raise RuntimeError("Authentication failed after retries")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                print(f"API request failed: {str(e)}")
                if retries >= self.max_retries:
                    raise
                retries += 1
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(f"API request failed after {self.max_retries} retries")

    def parse_response_data(
        self, fields: List[Dict[str, str]], data: List[List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse API response data based on field definitions

        Args:
            fields: Field definitions from API response
            data: Raw data rows

        Returns:
            List of parsed records
        """
        records = []

        for row in data:
            record = {}
            for i, field in enumerate(fields):
                value = row[i] if i < len(row) else None
                field_name = field["name"]
                field_type = field.get("dataType", "STRING")

                # Type conversion
                if value is None:
                    record[field_name] = None
                elif field_type == "DATETIME":
                    record[field_name] = value
                elif field_type == "BOOLEAN":
                    record[field_name] = bool(value)
                elif field_type in ("DOUBLE", "INTEGER"):
                    record[field_name] = float(value) if value else None
                else:
                    record[field_name] = value

            records.append(record)

        return records

    def fetch_paginated_data(
        self,
        endpoint: str,
        params: Dict[str, Any],
        api_type: str = "public",
        max_pages: Optional[int] = None,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Fetch paginated data from ERCOT API

        Args:
            endpoint: API endpoint path
            params: Query parameters
            api_type: "public" or "esr"
            max_pages: Maximum number of pages to fetch (None for all)

        Yields:
            List of records for each page
        """
        base_url = self.PUBLIC_BASE_URL if api_type == "public" else self.ESR_BASE_URL
        url = f"{base_url}{endpoint}"

        request_params = {
            **params,
            "size": self.page_size,
            "page": 1,
        }

        # Fetch first page
        print(f"Fetching page 1 from {endpoint}...")
        response = self.make_api_call(url, request_params, api_type)

        total_pages = response["_meta"]["totalPages"]
        total_records = response["_meta"]["totalRecords"]
        pages_to_fetch = min(max_pages, total_pages) if max_pages else total_pages

        print(f"Total records: {total_records}, Total pages: {total_pages}, Fetching: {pages_to_fetch}")

        # Yield first page
        first_page_data = self.parse_response_data(response["fields"], response["data"])
        yield first_page_data

        # Fetch remaining pages
        for page in range(2, pages_to_fetch + 1):
            print(f"Fetching page {page}/{pages_to_fetch}...")
            request_params["page"] = page
            page_response = self.make_api_call(url, request_params, api_type)
            page_data = self.parse_response_data(response["fields"], page_response["data"])
            yield page_data

    def fetch_lmp_by_settlement_point(
        self, from_date: datetime, max_pages: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Fetch LMP by Settlement Point data

        Args:
            from_date: Start datetime (UTC)
            max_pages: Maximum pages to fetch

        Yields:
            List of LMP records for each page
        """
        timestamp_str = from_date.strftime("%Y-%m-%dT%H:%M:%S")

        params = {
            "SCEDTimestampFrom": timestamp_str,
            "sort": "SCEDTimestamp",
            "dir": "asc",
        }

        yield from self.fetch_paginated_data(
            self.ENDPOINT_LMP_SETTLEMENT_POINT,
            params,
            api_type="public",
            max_pages=max_pages,
        )

    def fetch_spp_day_ahead_hourly(
        self, from_date: datetime, max_pages: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Fetch Day-Ahead Settlement Point Prices

        Args:
            from_date: Start date (UTC)
            max_pages: Maximum pages to fetch

        Yields:
            List of SPP records for each page
        """
        date_str = from_date.strftime("%Y-%m-%d")

        params = {
            "deliveryDateFrom": date_str,
            "sort": "deliveryDate",
            "dir": "asc",
        }

        yield from self.fetch_paginated_data(
            self.ENDPOINT_SPP_DAY_AHEAD,
            params,
            api_type="public",
            max_pages=max_pages,
        )


def create_client_from_env() -> ErcotClient:
    """
    Create ERCOT client from environment variables

    Required environment variables:
        - ERCOT_API_USERNAME
        - ERCOT_API_PASSWORD
        - ERCOT_PUBLIC_API_SUBSCRIPTION_KEY
        - ERCOT_ESR_API_SUBSCRIPTION_KEY (optional)

    Returns:
        Configured ErcotClient instance
    """
    return ErcotClient(
        username=os.environ["ERCOT_API_USERNAME"],
        password=os.environ["ERCOT_API_PASSWORD"],
        public_subscription_key=os.environ["ERCOT_PUBLIC_API_SUBSCRIPTION_KEY"],
        esr_subscription_key=os.environ.get("ERCOT_ESR_API_SUBSCRIPTION_KEY"),
    )
