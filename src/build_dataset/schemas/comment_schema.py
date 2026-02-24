from typing import TypedDict, Optional


class Comment(TypedDict):
    """
    Represents a single comment on a bug.
    """

    text: str
    attachment_id: Optional[int]
    creator: str
    creation_time: str


class BugComments(TypedDict):
    """
    Holds all comments for a single bug.
    """

    comments: list[Comment]


class BugCommentResponse(TypedDict):
    """
    API response for comments of multiple bugs.
    Keys are bug IDs as strings.
    """

    bugs: dict[str, BugComments]
