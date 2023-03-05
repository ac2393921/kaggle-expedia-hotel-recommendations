from typing import Optional

import pandas as pd
from loguru import logger

from src.dataset.shema import BASE_SCHEMA, DTYPES


def load_df_from_csv(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    logger.info(f"load dataframe from {file_path}")
    df = pd.read_csv(file_path, nrows=nrows, dtype=DTYPES)
    df = df.dropna()
    df = BASE_SCHEMA.validate(df)

    return df


def load_df_from_pickle(file_path: str) -> pd.DataFrame:
    logger.info(f"load dataframe from {file_path}")
    df: pd.DataFrame = pd.read_pickle(file_path)
    df = df.dropna()
    df = BASE_SCHEMA.validate(df)

    return df
