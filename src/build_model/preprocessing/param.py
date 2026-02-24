from config.config import config

TARGET_COL = "need_info_from_creator"
TEXTUAL_COLS = config.model_features_config.textual
CATEGORICAL_COLS = config.model_features_config.categorical

REQUIRED_COLS = [
    "id",
    "comments",
    "creator",
    "creation_time",
    "status",
    "summary",
    TARGET_COL,
]
