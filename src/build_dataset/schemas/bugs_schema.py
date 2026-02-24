from typing import TypedDict


class BugResponse(TypedDict):
    """
    Represents a bug returned from Bugzilla API.
    Fields depend on config.yaml definitions.
    """

    id: int
    creator: str
    creator_detail: object
    creation_time: str
    # Additional fields may be added depending on config.yaml


class BugSearchResponse(TypedDict):
    """
    Response from Bugzilla API for a bug search.
    """

    bugs: list[BugResponse]


class BugStored(TypedDict):
    """
    Represents a processed bug to store in dataset.
    """

    id: int
    creator: str
    creation_time: str
    need_info_from_creator: bool
    comments: str
    # Additional fields may be added depending on config.yaml
