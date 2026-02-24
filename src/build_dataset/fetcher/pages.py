import logging
import asyncio
from typing import Any
import aiohttp
from schemas.bugs_schema import BugResponse, BugSearchResponse
from schemas.history_schema import BugHistoryResponse, History
from schemas.comment_schema import BugCommentResponse, Comment
from utils.params import build_params
from config import config

logger = logging.getLogger(__name__)

BASE_URL = f"{config.bugzilla_config.base_url}/rest/bug"


async def _get_json(
    session: aiohttp.ClientSession, url: str, params: dict | None = None
) -> Any:
    """
    Helper to perform GET requests and return JSON.

    Raises:
        RuntimeError: If resp status < 400
    """
    async with session.get(url, params=params) as resp:
        if not resp.ok:
            logger.error("Error fetching %s: status %s", url, resp.status)
            raise RuntimeError(resp.status)
        return await resp.json()


async def fetch_page(
    session: aiohttp.ClientSession, offset: int, sem: asyncio.Semaphore
) -> list[BugResponse]:
    """
    Fetch a single page of bugs.
    """
    async with sem:
        data: BugSearchResponse = await _get_json(
            session, BASE_URL, params=build_params(offset)
        )
        bugs = data["bugs"]
        logger.info("Fetched offset %s -> %s bugs", offset, len(bugs))
        return bugs


async def fetch_history(
    session: aiohttp.ClientSession, bug_id: int, sem: asyncio.Semaphore
) -> list[History]:
    """
    Fetch /history for a single bug.
    """
    url = f"{BASE_URL}/{bug_id}/history"
    async with sem:
        data: BugHistoryResponse = await _get_json(session, url)
        return data["bugs"][0]["history"]


async def fetch_comment(
    session: aiohttp.ClientSession, bug_id: int, sem: asyncio.Semaphore
) -> list[Comment]:
    """
    Fetch /comment for a single bug.
    """
    url = f"{BASE_URL}/{bug_id}/comment"
    async with sem:
        data: BugCommentResponse = await _get_json(session, url)
        return data["bugs"][str(bug_id)]["comments"]
