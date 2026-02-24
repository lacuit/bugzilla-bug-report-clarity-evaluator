from typing import TypedDict


class Change(TypedDict):
    """
    Represents a single change in a bug's history.
    """

    field_name: str
    removed: str
    added: str
    attachment_id: int | None


class History(TypedDict):
    """
    Represents a history entry for a bug.
    """

    when: str
    who: str
    changes: list[Change]


class Bug(TypedDict):
    """
    Represents a bug including its history.
    """

    id: int
    alias: list[str]
    history: list[History]


class BugHistoryResponse(TypedDict):
    """
    API response for bug histories.
    """

    bugs: list[Bug]
