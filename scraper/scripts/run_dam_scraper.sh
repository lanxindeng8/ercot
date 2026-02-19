#!/bin/bash
# DAM LMP Scraper - runs every 15 minutes via launchd

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

# Run the scraper
cd src && "$PROJECT_DIR/venv/bin/python" scraper_dam_lmp.py

# Log completion
echo "$(date '+%Y-%m-%d %H:%M:%S') - DAM LMP scraper completed" >> "$PROJECT_DIR/logs/dam_scraper.log"
