"""
ERCOT CDR (Current Day Reports) Scraper

Fetches real-time LMP data directly from ERCOT's CDR HTML pages.
This provides near real-time data (within 5 minutes) vs the ~6 hour delay
from the Public API.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from html.parser import HTMLParser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# CDR URLs
CDR_RTM_LMP_URL = "https://www.ercot.com/content/cdr/html/current_np6788.html"

# ERCOT timezone
ERCOT_TIMEZONE = "America/Chicago"


class RTMLMPParser(HTMLParser):
    """HTML Parser for ERCOT RTM LMP CDR page"""

    def __init__(self):
        super().__init__()
        self.timestamp: Optional[str] = None
        self.records: List[Dict] = []
        self.current_row: List[str] = []
        self.in_td = False
        self.in_sched_time = False
        self.td_class = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == "div" and "schedTime" in attrs_dict.get("class", ""):
            self.in_sched_time = True
        elif tag == "td":
            self.in_td = True
            self.td_class = attrs_dict.get("class", "")

    def handle_endtag(self, tag):
        if tag == "div" and self.in_sched_time:
            self.in_sched_time = False
        elif tag == "td":
            self.in_td = False
        elif tag == "tr" and self.current_row:
            # Process completed row
            if len(self.current_row) >= 2:
                settlement_point = self.current_row[0].strip()
                lmp_str = self.current_row[1].strip()

                # Skip header rows and empty rows
                if settlement_point and lmp_str and settlement_point != "Settlement Point":
                    try:
                        lmp = float(lmp_str)
                        self.records.append({
                            "settlementPoint": settlement_point,
                            "lmp": lmp,
                        })
                    except ValueError:
                        pass  # Skip rows with non-numeric LMP

            self.current_row = []

    def handle_data(self, data):
        if self.in_sched_time:
            # Extract timestamp from "Last Updated: Feb 08, 2026 09:25:16"
            match = re.search(r"Last Updated:\s*(.+)", data)
            if match:
                self.timestamp = match.group(1).strip()
        elif self.in_td and "tdLeft" in self.td_class:
            self.current_row.append(data)


class CDRScraper:
    """Scraper for ERCOT CDR (Current Day Reports) pages"""

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize CDR scraper

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.timeout = timeout

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

    def fetch_rtm_lmp(self) -> Tuple[Optional[datetime], List[Dict]]:
        """
        Fetch real-time LMP data from CDR page

        Returns:
            Tuple of (timestamp, list of LMP records)
            Each record contains: settlementPoint, lmp
        """
        print(f"Fetching RTM LMP from CDR: {CDR_RTM_LMP_URL}")

        try:
            response = self.session.get(CDR_RTM_LMP_URL, timeout=self.timeout)
            response.raise_for_status()

            # Parse HTML
            parser = RTMLMPParser()
            parser.feed(response.text)

            # Parse timestamp
            timestamp = None
            if parser.timestamp:
                try:
                    # Format: "Feb 08, 2026 09:25:16"
                    timestamp = datetime.strptime(parser.timestamp, "%b %d, %Y %H:%M:%S")
                    print(f"CDR timestamp (CST): {timestamp}")
                except ValueError as e:
                    print(f"Failed to parse timestamp '{parser.timestamp}': {e}")

            print(f"Parsed {len(parser.records)} LMP records from CDR")

            return timestamp, parser.records

        except Exception as e:
            print(f"Error fetching CDR data: {e}")
            raise

    def close(self):
        """Close the HTTP session"""
        self.session.close()


def create_cdr_scraper() -> CDRScraper:
    """Create a CDR scraper instance"""
    return CDRScraper()


if __name__ == "__main__":
    # Test the scraper
    scraper = create_cdr_scraper()

    try:
        timestamp, records = scraper.fetch_rtm_lmp()

        print(f"\nTimestamp: {timestamp}")
        print(f"Total records: {len(records)}")

        # Print first 10 records
        print("\nFirst 10 records:")
        for record in records[:10]:
            print(f"  {record['settlementPoint']}: ${record['lmp']:.2f}")

        # Check for specific settlement points
        hub_points = [r for r in records if r['settlementPoint'].startswith('HB_')]
        lz_points = [r for r in records if r['settlementPoint'].startswith('LZ_')]
        print(f"\nHub points (HB_*): {len(hub_points)}")
        print(f"Load zone points (LZ_*): {len(lz_points)}")

    finally:
        scraper.close()
