#!/bin/bash
# RTM LMP CDR Scraper — real-time data from ERCOT CDR HTML pages
# Runs every 5 minutes. Does NOT use ERCOT API (no 429 risk).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${HOME}/.venvs/ercot-scraper/bin/python"

mkdir -p "$HOME/logs/trueflux"

cd "$PROJECT_DIR/src" && "$PYTHON" scraper_rtm_lmp_realtime.py

echo "$(date '+%Y-%m-%d %H:%M:%S') - RTM CDR scraper completed" >> "$HOME/logs/trueflux/rtm-lmp-cdr.log"
