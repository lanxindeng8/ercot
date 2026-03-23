#!/bin/bash
# RTM LMP API Scraper — historical backfill from ERCOT Public API
# Runs every hour. Uses ERCOT API (subject to 429 hourly bandwidth limit).
# Data has ~6 hour delay, so hourly frequency is sufficient.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${HOME}/.venvs/ercot-scraper/bin/python"

mkdir -p "$HOME/logs/trueflux"

cd "$PROJECT_DIR/src" && "$PYTHON" scraper_rtm_lmp.py

echo "$(date '+%Y-%m-%d %H:%M:%S') - RTM API scraper completed" >> "$HOME/logs/trueflux/rtm-lmp-api.log"
