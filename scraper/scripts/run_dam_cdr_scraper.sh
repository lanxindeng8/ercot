#!/bin/bash
# DAM LMP CDR Scraper - scrapes ERCOT CDR HTML page for DAM SPP data

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON="${HOME}/.venvs/ercot-scraper/bin/python"

mkdir -p "$HOME/logs/trueflux"

cd "$PROJECT_DIR/src" && "$PYTHON" scraper_dam_lmp_cdr.py

echo "$(date '+%Y-%m-%d %H:%M:%S') - DAM LMP CDR scraper completed" >> "$HOME/logs/trueflux/dam-lmp-cdr-scraper.log"
