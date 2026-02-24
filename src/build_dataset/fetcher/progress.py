import filecmp
import json
import logging
from pathlib import Path
import shutil

from config import CONFIG_PATH

logger = logging.getLogger(__name__)

PROGRESS_RUN_CONFIG_YAML = "progress_run_config.yaml"
WORKER_PROGRESS_COMBINED_JSON = "worker_progress_combined.json"


def save_progress_config(temp_dir: Path) -> None:
    """
    Copy the main config YAML to the temporary directory for tracking progress.
    """
    progress_config_path = temp_dir / PROGRESS_RUN_CONFIG_YAML
    shutil.copyfile(CONFIG_PATH, progress_config_path)


def can_restart_build_dataset(temp_dir: Path) -> bool:
    """
    Determine if the dataset build can be restarted from previous progress.
    Returns True if the saved YAML config matches the current CONFIG_PATH.
    """
    progress_config_path = temp_dir / PROGRESS_RUN_CONFIG_YAML
    if not progress_config_path.exists():
        return False
    return filecmp.cmp(progress_config_path, CONFIG_PATH, shallow=False)


def get_worker_progress(temp_dir: Path) -> list[int]:
    """
    Read combined worker progress and return offsets in order of worker IDs.
    """
    combined_file = temp_dir / WORKER_PROGRESS_COMBINED_JSON
    with combined_file.open("r") as f:
        worker_progress: dict[str, int] = json.load(f)

    worker_offsets: list[int] = [
        worker_progress[str(worker_id)]
        for worker_id in sorted(int(w) for w in worker_progress.keys())
    ]
    return worker_offsets


def save_worker_progress(temp_dir: Path, worker_id: int, last_offset: int) -> None:
    """
    Save progress for a single worker to its own JSON file.
    """
    file_path = temp_dir / f"{worker_id}_worker_progress.json"
    with file_path.open("w") as f:
        json.dump({str(worker_id): last_offset}, f)


def save_combined_worker_progress(temp_dir: Path) -> None:
    """
    Combine all individual worker progress files into a single JSON file.
    """
    combined: dict[str, int] = {}
    for file_path in temp_dir.glob("*_worker_progress.json"):
        with file_path.open("r") as f:
            data = json.load(f)
            combined.update(data)

    with (temp_dir / WORKER_PROGRESS_COMBINED_JSON).open("w") as f:
        json.dump(combined, f)


def refetch_worker_progress(temp_dir: Path) -> list[int] | None:
    """
    Attempt to resume previous worker progress.
    If the config has changed or no progress exists, reset the temp_dir and return None.
    """
    if can_restart_build_dataset(temp_dir):
        logger.info("Starting from previous progress...")
        return get_worker_progress(temp_dir)

    logger.info("Restarting progress from scratch (new YAML config or changed config)")
    shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    return None
