#!/bin/bash
# RTM LMP Scraper - runs every 5 minutes via launchd
# Runs both API scraper (for historical backfill) and CDR scraper (for real-time)

# Change to project directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Load environment variables from .env file
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"

# Run API scraper first (backfills historical data with ~6 hour delay)
echo "Running API scraper for historical backfill..."
cd src && "$PROJECT_DIR/venv/bin/python" scraper_rtm_lmp.py

# Run CDR scraper (real-time data within 5 minutes)
echo "Running CDR scraper for real-time data..."
cd "$PROJECT_DIR/src" && "$PROJECT_DIR/venv/bin/python" scraper_rtm_lmp_realtime.py

# Log completion
echo "$(date '+%Y-%m-%d %H:%M:%S') - RTM LMP scraper completed" >> "$PROJECT_DIR/logs/rtm_scraper.log"
