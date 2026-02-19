#!/usr/bin/env python3
"""
ERCOT Historical Data Downloader

Downloads historical data from ERCOT for the specified date range.
Products:
- NP4-190-CD: DAM Settlement Point Prices (daily)
- NP4-180-ER: Historical DAM Load Zone and Hub Prices (yearly)
- NP4-181-ER: Historical DAM Clearing Prices for Capacity (yearly)
- NP6-785-ER: Historical RTM Load Zone and Hub Prices (yearly)
- Fuel Mix Report: Wind generation data (2007-2024)

Usage:
    python download_historical.py --start-date 2015-12-01 --end-date 2016-01-31
"""

import os
import sys
import json
import time
import re
import requests
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# ERCOT API endpoints
ERCOT_DOC_LIST_API = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
ERCOT_DOWNLOAD_URL = "https://www.ercot.com/misdownload/servlets/mirDownload"

# Direct download URLs for static files
FUEL_MIX_URL = "https://www.ercot.com/files/docs/2021/03/10/FuelMixReport_PreviousYears.zip"

# Product configurations with numeric report type IDs
# These IDs are found on the product pages (reportTypeId_i)
PRODUCTS = {
    "NP4-190-CD": {
        "name": "DAM Settlement Point Prices",
        "description": "All Resource Nodes, Load Zones, Trading Hubs (daily)",
        "report_type_id": "12331",
        "type": "daily",
    },
    "NP4-180-ER": {
        "name": "Historical DAM Load Zone and Hub Prices",
        "description": "Annual Hub/Zone DAM prices (DAMLZHBSPP)",
        "report_type_id": "13060",
        "type": "yearly",
    },
    "NP4-181-ER": {
        "name": "Historical DAM Clearing Prices for Capacity",
        "description": "DAM capacity clearing prices (DAMASMCPC)",
        "report_type_id": "13091",
        "type": "yearly",
    },
    "NP6-785-ER": {
        "name": "Historical RTM Load Zone and Hub Prices",
        "description": "Annual Hub/Zone RTM prices (RTMLZHBSPP)",
        "report_type_id": "13061",
        "type": "yearly",
    },
    "FUEL_MIX": {
        "name": "Fuel Mix Report 2007-2024",
        "description": "Wind and other generation by fuel type",
        "type": "static",
        "url": FUEL_MIX_URL,
    },
}


class ERCOTDownloader:
    def __init__(self, output_dir: str, start_date: str, end_date: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })

    def download_file(self, url: str, filename: str, subdir: str = "") -> bool:
        """Download a file from URL to output directory."""
        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir

        filepath = save_dir / filename

        if filepath.exists():
            print(f"  [SKIP] Already exists: {filename}")
            return True

        try:
            print(f"  [DOWN] Downloading: {filename}")
            response = self.session.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Check if it's an error response
            content_type = response.headers.get('content-type', '')
            if 'xml' in content_type and response.content.startswith(b'<?xml'):
                print(f"  [ERROR] Server returned error for {filename}")
                return False

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\r  [DOWN] {filename}: {pct:.1f}%", end="", flush=True)

            size_mb = downloaded / 1024 / 1024
            print(f"\r  [DONE] {filename} ({size_mb:.1f} MB)")
            return True

        except Exception as e:
            print(f"\n  [ERROR] Failed to download {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False

    def download_fuel_mix(self) -> bool:
        """Download Fuel Mix Report (contains wind generation data)."""
        print("\n" + "="*60)
        print("Downloading Fuel Mix Report (Wind Generation Data)")
        print("="*60)

        return self.download_file(
            FUEL_MIX_URL,
            "FuelMixReport_2007-2024.zip",
            "fuel_mix"
        )

    def get_years_in_range(self) -> List[int]:
        """Get list of years covered by the date range."""
        years = set()
        current = self.start_date
        while current <= self.end_date:
            years.add(current.year)
            current += timedelta(days=365)
        years.add(self.end_date.year)
        return sorted(years)

    def get_document_list(self, report_type_id: str) -> List[Dict]:
        """Get list of documents from ERCOT API."""
        url = f"{ERCOT_DOC_LIST_API}?reportTypeId={report_type_id}"

        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            doc_list = data.get("ListDocsByRptTypeRes", {}).get("DocumentList", [])
            documents = []

            for item in doc_list:
                doc = item.get("Document", {})
                documents.append({
                    "doc_id": doc.get("DocID"),
                    "friendly_name": doc.get("FriendlyName"),
                    "constructed_name": doc.get("ConstructedName"),
                    "publish_date": doc.get("PublishDate"),
                    "extension": doc.get("Extension"),
                    "size": int(doc.get("ContentSize", 0)),
                })

            return documents

        except Exception as e:
            print(f"  [ERROR] Failed to get document list: {e}")
            return []

    def download_yearly_product(self, product_id: str) -> bool:
        """Download yearly archive files for a product."""
        product = PRODUCTS[product_id]
        print(f"\n" + "="*60)
        print(f"Downloading {product_id}: {product['name']}")
        print("="*60)

        report_type_id = product.get("report_type_id")
        if not report_type_id:
            print(f"  [ERROR] No report type ID for {product_id}")
            return False

        years = self.get_years_in_range()
        print(f"Years to download: {years}")

        # Get document list from API
        print(f"  Fetching document list from ERCOT...")
        documents = self.get_document_list(report_type_id)

        if not documents:
            print(f"  [WARN] No documents found")
            return False

        print(f"  Found {len(documents)} total documents")

        success_count = 0
        product_dir = self.output_dir / product_id
        product_dir.mkdir(parents=True, exist_ok=True)

        for year in years:
            # Find document for this year
            year_str = str(year)
            matching_docs = [d for d in documents if year_str in d.get("friendly_name", "")]

            if matching_docs:
                doc = matching_docs[0]  # Get the first/latest for this year
                doc_id = doc["doc_id"]
                friendly_name = doc["friendly_name"]
                ext = doc.get("extension", "zip")

                download_url = f"{ERCOT_DOWNLOAD_URL}?doclookupId={doc_id}"
                filename = f"{product_id}_{friendly_name}.{ext}"

                if self.download_file(download_url, filename, product_id):
                    success_count += 1
                else:
                    print(f"  [WARN] Failed to download {year}")

                # Rate limiting
                time.sleep(1)
            else:
                print(f"  [WARN] No document found for {year}")

        return success_count > 0

    def get_report_type_id_from_page(self, product_id: str) -> Optional[str]:
        """Scrape report type ID from ERCOT product page."""
        url = f"https://www.ercot.com/mp/data-products/data-product-details?id={product_id}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Look for reportTypeId in the JavaScript
            match = re.search(r'reportTypeId["\s:_=]+["\']?(\d+)', response.text)
            if match:
                return match.group(1)

            # Also try reportTypeId_i pattern
            match = re.search(r'"reportTypeId_i"\s*:\s*"?(\d+)"?', response.text)
            if match:
                return match.group(1)

        except Exception as e:
            print(f"  Could not get report type ID from page: {e}")

        return None

    def download_daily_product(self, product_id: str) -> bool:
        """Download daily data files for a product (like NP4-190-CD)."""
        product = PRODUCTS[product_id]
        print(f"\n" + "="*60)
        print(f"Downloading {product_id}: {product['name']}")
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        print("="*60)

        # First get the report type ID
        report_type_id = product.get("report_type_id")
        if not report_type_id:
            print(f"  Getting report type ID from ERCOT website...")
            report_type_id = self.get_report_type_id_from_page(product_id)
            if report_type_id:
                print(f"  Found report type ID: {report_type_id}")
            else:
                print(f"  [ERROR] Could not find report type ID")
                return False

        # Get document list
        print(f"  Fetching document list...")
        documents = self.get_document_list(report_type_id)

        if not documents:
            print(f"  [WARN] No documents found")
            return False

        print(f"  Found {len(documents)} total documents")

        # Filter by date range
        start_str = self.start_date.strftime("%Y%m%d")
        end_str = self.end_date.strftime("%Y%m%d")

        matching_docs = []
        for doc in documents:
            # Extract date from constructed name or friendly name
            name = doc.get("constructed_name", "") or doc.get("friendly_name", "")
            # Look for date pattern YYYYMMDD
            date_match = re.search(r'\.(\d{8})\.', name)
            if date_match:
                doc_date = date_match.group(1)
                if start_str <= doc_date <= end_str:
                    doc["date"] = doc_date
                    matching_docs.append(doc)

        print(f"  {len(matching_docs)} documents in date range")

        if not matching_docs:
            return False

        success_count = 0
        product_dir = self.output_dir / product_id
        product_dir.mkdir(parents=True, exist_ok=True)

        for i, doc in enumerate(matching_docs):
            doc_id = doc["doc_id"]
            doc_date = doc.get("date", "unknown")
            ext = doc.get("extension", "zip")

            download_url = f"{ERCOT_DOWNLOAD_URL}?doclookupId={doc_id}"
            filename = f"{product_id}_{doc_date}.{ext}"

            if self.download_file(download_url, filename, product_id):
                success_count += 1

            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(matching_docs)}")

            # Rate limiting
            time.sleep(0.5)

        return success_count > 0

    def download_all(self) -> Dict[str, bool]:
        """Download all products."""
        results = {}

        # 1. Download Fuel Mix (static file)
        results["FUEL_MIX"] = self.download_fuel_mix()

        # 2. Download yearly products
        for product_id in ["NP4-180-ER", "NP4-181-ER", "NP6-785-ER"]:
            # First try to get report type ID if not known
            if not PRODUCTS[product_id].get("report_type_id"):
                print(f"\n  Getting report type ID for {product_id}...")
                rtid = self.get_report_type_id_from_page(product_id)
                if rtid:
                    PRODUCTS[product_id]["report_type_id"] = rtid
                    print(f"  Found: {rtid}")

            results[product_id] = self.download_yearly_product(product_id)

        # 3. Download daily products
        for product_id in ["NP4-190-CD"]:
            results[product_id] = self.download_daily_product(product_id)

        return results

    def print_summary(self, results: Dict[str, bool]):
        """Print download summary."""
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)

        for product_id, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            product_name = PRODUCTS[product_id]["name"]
            print(f"  {product_id}: {status}")
            print(f"    {product_name}")

        print(f"\nFiles saved to: {self.output_dir}")

        # List downloaded files
        print("\nDownloaded files:")
        for subdir in self.output_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                print(f"  {subdir.name}/: {len(files)} files, {total_size/1024/1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download ERCOT historical data")
    parser.add_argument(
        "--start-date",
        default="2015-12-01",
        help="Start date (YYYY-MM-DD), default: 2015-12-01"
    )
    parser.add_argument(
        "--end-date",
        default="2016-01-31",
        help="End date (YYYY-MM-DD), default: 2016-01-31"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: historical_data in script dir)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "historical_data"

    print("="*60)
    print("ERCOT Historical Data Downloader")
    print("="*60)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output: {output_dir}")
    print()

    downloader = ERCOTDownloader(
        output_dir=str(output_dir),
        start_date=args.start_date,
        end_date=args.end_date
    )

    results = downloader.download_all()
    downloader.print_summary(results)


if __name__ == "__main__":
    main()
