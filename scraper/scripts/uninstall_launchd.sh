#!/bin/bash
# Uninstall launchd services for ERCOT scrapers on macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Uninstalling ERCOT Scraper Services ===${NC}"

LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

# Unload and remove RTM service
if [ -f "$LAUNCH_AGENTS_DIR/com.trueflux.rtm-lmp-scraper.plist" ]; then
    echo "Unloading RTM LMP scraper..."
    launchctl unload "$LAUNCH_AGENTS_DIR/com.trueflux.rtm-lmp-scraper.plist" 2>/dev/null || true
    rm "$LAUNCH_AGENTS_DIR/com.trueflux.rtm-lmp-scraper.plist"
    echo -e "${GREEN}RTM LMP scraper removed${NC}"
else
    echo "RTM LMP scraper not installed"
fi

# Unload and remove DAM service
if [ -f "$LAUNCH_AGENTS_DIR/com.trueflux.dam-lmp-scraper.plist" ]; then
    echo "Unloading DAM LMP scraper..."
    launchctl unload "$LAUNCH_AGENTS_DIR/com.trueflux.dam-lmp-scraper.plist" 2>/dev/null || true
    rm "$LAUNCH_AGENTS_DIR/com.trueflux.dam-lmp-scraper.plist"
    echo -e "${GREEN}DAM LMP scraper removed${NC}"
else
    echo "DAM LMP scraper not installed"
fi

echo ""
echo -e "${GREEN}=== Uninstallation Complete ===${NC}"
