import polars as pl
import logging

logger = logging.getLogger(__name__)


def encode_categorical(df: pl.DataFrame, categorical_cols: list[str]):
    """
    Perform one-hot encoding on categorical columns.
    """
    encoded_cols = []

    for col in categorical_cols:
        df = df.with_columns(pl.col(col).cast(pl.Categorical))
        ohe_df = df.select([col]).to_dummies()
        encoded_cols.extend(ohe_df.columns)
        df = df.hstack(ohe_df)

    return df, encoded_cols
