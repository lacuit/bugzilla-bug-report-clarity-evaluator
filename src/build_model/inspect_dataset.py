from typing import cast

import polars as pl
from config.config import global_config, GLOBAL_ROOT
from data.load_data import load_dataset

OUTPUT_PATH_PARQUET = global_config.dataset_store_config.output_path_parquet


def main():
    lf = pl.scan_parquet(GLOBAL_ROOT / OUTPUT_PATH_PARQUET)

    # Drop duplicates
    lf_unique = lf.unique(subset="id", keep="any")
    print(lf_unique.head().collect())

    print(cast(pl.DataFrame, lf.select(pl.len()).collect()).item())
    print(cast(pl.DataFrame, lf_unique.select(pl.len()).collect()).item())
    print(load_dataset().height)

    # Value Counts
    for col in lf_unique.collect_schema().names():
        vc_df = lf_unique.select(pl.col(col).value_counts(sort=False)).collect()
        print(vc_df)

    print(lf.collect_schema().names())


if __name__ == "__main__":
    main()
