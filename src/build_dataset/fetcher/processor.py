from datetime import datetime
from typing import cast
from schemas.bugs_schema import BugResponse, BugStored
from schemas.history_schema import History
from schemas.comment_schema import Comment
from utils.params import INTERESTED_FIELDS
from config import config

COMMENT_SEPARATOR = config.dataset_store_config.comment_separator
USE_NEED_INFO_PROXY = config.bugzilla_config.use_need_info_proxy


def detect_needinfo(bug: BugResponse, history: list[History]) -> datetime | None:
    """
    Detect if a needinfo flag was set for the bug creator in history.
    Returns the last timestamp when it occurred, or None if not found.
    """
    needinfo_timestamp: datetime | None = None

    for change_log in history:
        for change in change_log["changes"]:
            if (
                change["field_name"] == "flagtypes.name"
                and change_log["who"] != bug["creator"]
                and change["added"].startswith("needinfo?")
                and bug["creator"] in change["added"]
            ):
                needinfo_timestamp = datetime.fromisoformat(change_log["when"])

    return needinfo_timestamp


def detect_needinfo_proxy(bug: BugResponse, comments: list[Comment]) -> datetime | None:
    """
    Proxy detection for non-Mozilla Bugzilla:
    Returns timestamp of the first comment containing '?' by a non-creator,
    only if the previous comment was by the creator.
    """
    previous_was_creator = False

    for comment in comments:
        if comment["creator"] == bug["creator"]:
            previous_was_creator = True
        elif "?" in comment["text"] and previous_was_creator:
            return datetime.fromisoformat(comment["creation_time"])
        else:
            previous_was_creator = False

    return None


def revert_fields_after_timestamp(
    bug: BugResponse, history: list[History], timestamp: datetime
) -> None:
    """
    Revert bug fields in INTERESTED_FIELDS to their values before a given timestamp.
    """
    for change_log in reversed(history):
        change_time = datetime.fromisoformat(change_log["when"])
        if change_time >= timestamp:
            for change in change_log["changes"]:
                if change["field_name"] in INTERESTED_FIELDS:
                    cast(dict, bug)[change["field_name"]] = change["removed"]


def merge_creator_comments(bug: BugResponse, comments: list[Comment]) -> str:
    """
    Merge all comments from the bug creator into a single string.
    """
    return COMMENT_SEPARATOR.join(
        c["text"] for c in comments if c["creator"] == bug["creator"]
    )


def process_bug(
    bug: BugResponse, history: list[History], comments: list[Comment]
) -> BugStored:
    """
    Process a bug, optionally using needinfo proxy for non-Mozilla Bugzilla.
    Reverts fields after needinfo timestamp and merges creator comments.
    """
    bug_copy = bug.copy()

    needinfo_timestamp: datetime | None
    if USE_NEED_INFO_PROXY:
        needinfo_timestamp = detect_needinfo_proxy(bug_copy, comments)
    else:
        needinfo_timestamp = detect_needinfo(bug_copy, history)

    if needinfo_timestamp:
        revert_fields_after_timestamp(bug_copy, history, needinfo_timestamp)

    return {k: v for k, v in bug_copy.items() if k != "creator_detail"} | {
        "comments": merge_creator_comments(bug_copy, comments),
        "need_info_from_creator": bool(needinfo_timestamp),
    }
