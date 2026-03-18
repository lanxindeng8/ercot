"""Data fetching and feature engineering modules"""


def __getattr__(name):
    if name in ("InfluxDBFetcher", "create_fetcher_from_env"):
        from .influxdb_fetcher import InfluxDBFetcher, create_fetcher_from_env
        return {"InfluxDBFetcher": InfluxDBFetcher, "create_fetcher_from_env": create_fetcher_from_env}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
