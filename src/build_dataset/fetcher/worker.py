import asyncio
import logging
from pathlib import Path
from typing import Optional
from asyncio import Semaphore, TaskGroup

import aiohttp
from fetcher.client import exponential_backoff_retry
from fetcher.pages import fetch_page, fetch_history, fetch_comment
from fetcher.processor import process_bug
from fetcher.progress import (
    refetch_worker_progress,
    save_combined_worker_progress,
    save_progress_config,
    save_worker_progress,
)
from schemas.history_schema import History
from schemas.comment_schema import Comment
from utils.parquet import save_batch
from schemas.bugs_schema import BugResponse, BugStored
from config import config
import traceback

logger = logging.getLogger(__name__)

WORKER_CONCURRENCY = config.api_fetch_config.worker_concurrency_limit
PAGES_PER_BATCH = config.api_fetch_config.pages_per_batch
LIMIT = config.api_fetch_config.page_size
PARALLEL_REQUESTS_MAX = config.api_fetch_config.parallel_requests_limit


async def fetch_bug_details(
    session: aiohttp.ClientSession, bug: BugResponse, sem: Semaphore
) -> tuple[BugResponse, list[History], list[Comment]]:
    """
    Fetch history and comments for a single bug.
    """
    async with TaskGroup() as tg:
        history_task = tg.create_task(fetch_history(session, bug["id"], sem))
        comment_task = tg.create_task(fetch_comment(session, bug["id"], sem))
    return bug, history_task.result(), comment_task.result()


async def worker(
    session: aiohttp.ClientSession,
    worker_id: int,
    temp_dir: Path,
    sem: Semaphore,
    starting_offset: Optional[int],
) -> None:
    """
    Worker that fetches pages of bugs, fetches details, processes them,
    and saves progress in batches.
    """
    batch_rows: list[BugStored] = []
    batch_number = 0
    pages_fetched = 0
    offset = starting_offset if starting_offset is not None else worker_id * LIMIT

    while True:
        try:
            bugs = await fetch_page(session, offset, sem)
            if not bugs:
                break  # No more pages

            # Fetch history + comments concurrently
            async with TaskGroup() as tg:
                tasks = [
                    tg.create_task(fetch_bug_details(session, bug, sem)) for bug in bugs
                ]

            results = [t.result() for t in tasks]

            for bug, history, comments in results:
                batch_rows.append(process_bug(bug, history, comments))

            pages_fetched += 1

            # Save batch if enough pages collected
            if pages_fetched >= PAGES_PER_BATCH:
                save_batch(batch_rows, temp_dir, worker_id, batch_number)
                batch_rows.clear()
                pages_fetched = 0
                batch_number += 1

            # Move to next stride
            offset += WORKER_CONCURRENCY * LIMIT

        except Exception as e:
            logger.error("Worker %s error at offset %s: %s", worker_id, offset, e)
            logger.info("Rerun script when completed to restart ONLY this worker")
            logger.info(traceback.format_exc())
            break

    # Save last offset and flush remaining batch
    save_worker_progress(temp_dir, worker_id, offset)
    if batch_rows:
        save_batch(batch_rows, temp_dir, worker_id, batch_number)


async def fetch_all(temp_dir: Path, output_parquet: Path) -> None:
    """
    Entry point to fetch all bugs using multiple workers.
    Resumes previous progress if available.

    Raises:
        FileExistsError: If existing database will be replaced as its not a continuing run
    """
    worker_starting_offsets = refetch_worker_progress(temp_dir)

    if worker_starting_offsets is None and output_parquet.exists():
        logger.error(
            "Cannot generate a new dataset as it will overwrite existing dataset"
        )
        logger.info(f"Please move/remove before runnning again at {output_parquet}")
        raise FileExistsError("Dataset will be replaced")

    sem = asyncio.Semaphore(PARALLEL_REQUESTS_MAX)

    async with aiohttp.ClientSession(
        middlewares=[exponential_backoff_retry]
    ) as session:
        async with TaskGroup() as tg:
            for worker_id in range(WORKER_CONCURRENCY):
                offset = None
                if worker_starting_offsets is not None:
                    offset = worker_starting_offsets[worker_id]

                tg.create_task(worker(session, worker_id, temp_dir, sem, offset))

    save_combined_worker_progress(temp_dir)
    save_progress_config(temp_dir)
