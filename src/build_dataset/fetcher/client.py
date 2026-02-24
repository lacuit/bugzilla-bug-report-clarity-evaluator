import asyncio
import logging
import aiohttp
from config import config

logger = logging.getLogger(__name__)

RETRY_MAX_ATTEMPT = config.api_fetch_config.retry_max_attempt
BASE_RETRY_DELAY_S = config.api_fetch_config.base_retry_delay_s


async def exponential_backoff_retry(
    req: aiohttp.ClientRequest, handler: aiohttp.ClientHandlerType
) -> aiohttp.ClientResponse:
    """
    Retry an aiohttp request using exponential backoff.
    Returns the response if successful, otherwise returns the last response after retries.
    """
    for attempt in range(1, RETRY_MAX_ATTEMPT + 1):
        resp = await handler(req)
        if resp.ok:
            return resp

        if attempt < RETRY_MAX_ATTEMPT:
            delay = BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
            logger.debug(
                "Request failed (attempt %s/%s). Retrying in %.2f seconds...",
                attempt,
                RETRY_MAX_ATTEMPT,
                delay,
            )
            await asyncio.sleep(delay)

    logger.warning("Request failed after %s attempts", RETRY_MAX_ATTEMPT)
    return resp
