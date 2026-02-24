from dataclasses import dataclass
from typing import Optional


# --- Build Dataset Config --- #
@dataclass
class BugzillaConfig:
    """
    Configuration for Bugzilla API.
    """

    base_url: str
    use_need_info_proxy: bool


@dataclass
class SearchConfig:
    """
    Configuration for bug search parameters.
    """

    lookback_days: int
    interested_fields: list[str]
    product: Optional[str] = None


@dataclass
class APIFetchConfig:
    """
    Configuration for API fetch behavior.
    """

    page_size: int
    worker_concurrency_limit: int
    pages_per_batch: int
    parallel_requests_limit: int
    retry_max_attempt: int
    base_retry_delay_s: int


@dataclass
class DatasetStoreConfig:
    """
    Configuration for dataset storage.
    """

    store_dir_temp: str
    comment_separator: str


@dataclass
class LogConfig:
    """
    Logging configuration.
    """

    level: str


@dataclass
class Config:
    """
    Main configuration.
    """

    bugzilla_config: BugzillaConfig
    search_config: SearchConfig
    api_fetch_config: APIFetchConfig
    dataset_store_config: DatasetStoreConfig
    log_config: LogConfig


# --- Global Config --- #
@dataclass
class GlobalDatasetStoreConfig:
    """Global dataset storage configuration."""

    output_path_parquet: str


@dataclass
class GlobalConfig:
    """Global configuration."""

    dataset_store_config: GlobalDatasetStoreConfig
