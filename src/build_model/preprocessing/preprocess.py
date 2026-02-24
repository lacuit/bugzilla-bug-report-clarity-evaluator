import numpy as np
import logging
import polars as pl
from preprocessing.categorical_encoding import encode_categorical
from preprocessing.embeddings import generate_embeddings
from preprocessing.param import (
    CATEGORICAL_COLS,
    REQUIRED_COLS,
    TARGET_COL,
    TEXTUAL_COLS,
)

logger = logging.getLogger(__name__)


def check_enough_features():
    """
    Checks if at least one feature is used in model
    Raises:
        ValueError: If less than one feature
    """
    model_features = CATEGORICAL_COLS + TEXTUAL_COLS
    if len(model_features) < 1:
        raise ValueError("Add at least one feature in config.yaml")


def check_features_exist(df: pl.DataFrame):
    """
    Checks if all features configured in config.yaml exists
    Raises:
        KeyError: If required columns are missing.
    """

    for col_name in set(CATEGORICAL_COLS + [TARGET_COL] + TEXTUAL_COLS + REQUIRED_COLS):
        if col_name not in df.collect_schema().names():
            raise KeyError(
                f"Column '{col_name}' configured in config.yaml not found in DataFrame"
            )


def prepare_features(df: pl.DataFrame):
    """
    Preprocess dataset to produce features and target for model training.
    Raises:
        KeyError: If required columns are missing.
    """
    check_enough_features()
    check_features_exist(df)

    # Compute scale_pos_weight
    n_pos = df.filter(df[TARGET_COL]).height
    n_neg = df.filter(~df[TARGET_COL]).height
    scale_pos_weight = n_neg / n_pos
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Fill missing text
    fill_exprs = [df[col].fill_null("") for col in TEXTUAL_COLS]
    df = df.with_columns(fill_exprs)

    # Target encode categorical columns
    df, encoded_cols = encode_categorical(df, CATEGORICAL_COLS)

    # Generate embeddings
    embeddings_list = [generate_embeddings(df[col].to_list()) for col in TEXTUAL_COLS]

    # Combine features
    X_cat = df[encoded_cols].to_numpy().astype(np.float32)

    arrays_to_stack = embeddings_list.copy()

    # categorical feature
    if X_cat.shape[0] > 0 and X_cat.shape[1] > 0:
        arrays_to_stack.append(X_cat)

    X = np.hstack(arrays_to_stack)
    y = df[TARGET_COL].to_numpy()

    logger.info(f"Final feature shape: {X.shape}")
    return X, y, scale_pos_weight
