"""Data fetching and feature engineering modules"""

import importlib as _importlib


def __getattr__(name):
    if name in ("InfluxDBFetcher", "create_fetcher_from_env"):
        from .influxdb_fetcher import InfluxDBFetcher, create_fetcher_from_env
        return {"InfluxDBFetcher": InfluxDBFetcher, "create_fetcher_from_env": create_fetcher_from_env}[name]
    # Allow sub-package discovery (weather, ercot) by Python's import system
    try:
        return _importlib.import_module(f".{name}", __name__)
    except ImportError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
