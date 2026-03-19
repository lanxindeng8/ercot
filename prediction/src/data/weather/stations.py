"""
Zone-to-city weather station mapping for ERCOT zones.

Each station represents a major city whose weather is representative
of one or more ERCOT settlement zones.
"""

WEATHER_STATIONS = {
    "san_antonio": {"lat": 29.42, "lon": -98.49, "zones": ["LZ_CPS"]},
    "midland": {"lat": 31.95, "lon": -102.18, "zones": ["LZ_WEST", "HB_WEST"]},
    "houston": {"lat": 29.76, "lon": -95.37, "zones": ["LZ_HOUSTON", "HB_HOUSTON"]},
    "dallas": {"lat": 32.78, "lon": -96.80, "zones": ["HB_NORTH", "LZ_NORTH"]},
    "corpus_christi": {"lat": 27.80, "lon": -97.40, "zones": ["HB_SOUTH", "LZ_SOUTH"]},
    "austin": {
        "lat": 30.27,
        "lon": -97.74,
        "zones": ["HB_BUSAVG", "HB_HUBAVG", "HB_PAN", "LZ_AEN", "LZ_LCRA", "LZ_RAYBN"],
    },
}

# Reverse lookup: zone -> station name
_ZONE_TO_STATION = {}
for _station, _info in WEATHER_STATIONS.items():
    for _zone in _info["zones"]:
        _ZONE_TO_STATION[_zone] = _station


def get_station_for_zone(zone: str) -> str:
    """Return the weather station name for a given ERCOT zone.

    Raises KeyError if the zone is not mapped.
    """
    return _ZONE_TO_STATION[zone]
