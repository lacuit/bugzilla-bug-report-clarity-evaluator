import logging
from pathlib import Path
from typing import cast
import polars as pl

from schemas.bugs_schema import BugStored

logger = logging.getLogger(__name__)


def save_batch(
    batch_rows: list[BugStored], temp_dir: Path, worker_id: int, batch_number: int
) -> None:
    """
    Save a batch of bug rows to a Parquet file.
    """
    if not batch_rows:
        return

    batch_file = temp_dir / f"worker_{worker_id}_batch_{batch_number}.parquet"
    try:
        pl.from_dicts(batch_rows).write_parquet(batch_file)
        logger.info(
            "Worker %s: saved %s rows to %s", worker_id, len(batch_rows), batch_file
        )
    except Exception as e:
        logger.error(
            "Worker %s failed to save batch %s: %s", worker_id, batch_number, e
        )


def merge_parquets(temp_dir: Path, output_parquet: Path) -> None:
    """
    Merge all Parquet batch files in a directory into a single Parquet file.
    """
    parquet_files = sorted(temp_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No Parquet files found to merge.")
        return

    # Load lazily for streaming merge
    lazy_frames = [pl.scan_parquet(f) for f in parquet_files]

    # Ensure consistent column order
    columns = lazy_frames[0].collect_schema().names()
    ref_columns = ["id"] + sorted(c for c in columns if c != "id")
    lazy_frames_aligned = [lf.select(ref_columns) for lf in lazy_frames]

    merged = pl.concat(lazy_frames_aligned)
    df = cast(pl.DataFrame, merged.collect(engine="streaming"))

    df.write_parquet(output_parquet)
    logger.info("Merged %s files into %s", len(parquet_files), output_parquet)
