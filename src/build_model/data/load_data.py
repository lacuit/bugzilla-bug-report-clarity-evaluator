from datetime import timedelta
from typing import cast
import polars as pl
from config.config import GLOBAL_ROOT, global_config


def load_dataset() -> pl.DataFrame:
    """
    Load the dataset from the configured parquet path and filter rows older than 14 days
    from the latest creation_time. This removes unclear bugs that may not have been triaged yet.

    Raises:
        FileNotFoundError: If the parquet file does not exist at the specified path.
    """
    # Lazy scan of parquet
    lf = pl.scan_parquet(
        GLOBAL_ROOT / global_config.dataset_store_config.output_path_parquet
    )

    # Filter rows older than 14 days using temporary __creation_time
    lf_filtered = (
        lf.with_columns(
            pl.col("creation_time")
            .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ")
            .alias("__creation_time")
        )
        .unique(subset="id", keep="any")
        .filter(
            pl.col("__creation_time")
            <= (pl.col("__creation_time").max() - timedelta(days=14))
        )
    )

    df = lf_filtered.drop("__creation_time").collect()

    return cast(pl.DataFrame, df)
