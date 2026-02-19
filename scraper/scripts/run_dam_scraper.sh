#!/bin/bash
# DAM LMP Scraper - runs every 15 minutes via launchd

# Change to project directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${HOME}/.venvs/ercot-scraper/bin/python"

# Ensure logs directory exists
mkdir -p "$HOME/logs/trueflux"

# Run the scraper
cd "$PROJECT_DIR/src" && "$PYTHON" scraper_dam_lmp.py

# Log completion
echo "$(date '+%Y-%m-%d %H:%M:%S') - DAM LMP scraper completed" >> "$HOME/logs/trueflux/dam-lmp-scraper.log"
