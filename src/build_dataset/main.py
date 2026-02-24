import asyncio
import logging
from pathlib import Path

from fetcher.worker import fetch_all
from utils.params import CREATION_TIME
from utils.parquet import merge_parquets
from config import GLOBAL_ROOT, PROJECT_ROOT, config, global_config

# --- Config / Paths --- #
STORE_DIR_TEMP = config.dataset_store_config.store_dir_temp
OUTPUT_PATH_PARQUET = global_config.dataset_store_config.output_path_parquet
LOG_LEVEL = config.log_config.level

# --- Logging setup --- #
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Main entry point for fetching Bugzilla data and saving to Parquet.
    """
    logger.info("Starting data fetch")

    # Temporary storage directory per run (based on creation date)
    temp_dir = PROJECT_ROOT / Path(STORE_DIR_TEMP) / CREATION_TIME.replace("-", "_")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Output Parquet path
    output_parquet = GLOBAL_ROOT / Path(OUTPUT_PATH_PARQUET)

    # Fetch all bugs concurrently
    await fetch_all(temp_dir, output_parquet)

    # Merge all batch Parquet files into a single output file
    merge_parquets(temp_dir, output_parquet)
    logger.info(f"Data fetch complete. Output written to {output_parquet}")


if __name__ == "__main__":
    logger.info(f"Log level set to {LOG_LEVEL}")
    asyncio.run(main())
