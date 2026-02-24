from datetime import datetime, timedelta
from typing import Any
from config import config

# Base fields always included
BASE_FIELDS = ["id", "creator", "creation_time", "status"]
INTERESTED_FIELDS = config.search_config.interested_fields
LIMIT = config.api_fetch_config.page_size
LOOKBACK_DAYS_WINDOW = config.search_config.lookback_days


def _compute_creation_date_start() -> str:
    """
    Compute the start date for bug creation filter based on lookback window.
    """
    return (datetime.now() - timedelta(days=LOOKBACK_DAYS_WINDOW)).date().isoformat()


# Create once at module load to prevent rollover during fetch
CREATION_TIME = _compute_creation_date_start()


def build_params(offset: int) -> dict[str, Any]:
    """
    Build query parameters for Bugzilla API requests.
    """
    params: dict[str, Any] = {
        "creation_time": CREATION_TIME,
        "limit": LIMIT,
        "offset": offset,
        "include_fields": list(dict.fromkeys(BASE_FIELDS + INTERESTED_FIELDS)),
    }

    product = config.search_config.product
    if product:
        params["product"] = product

    return params
