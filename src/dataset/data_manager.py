from typing import Optional

import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import DataFrame

from src.dataset.shema import DTYPES, BaseSchema


@pa.check_types
def load_df_from_csv(
    file_path: str, nrows: Optional[int] = None
) -> DataFrame[BaseSchema]:
    logger.info(f"load dataframe from {file_path}")
    df = pd.read_csv(file_path, nrows=nrows, dtype=DTYPES)
    df = df.dropna()

    return df


@pa.check_types
def load_df_from_pickle(file_path: str) -> DataFrame[BaseSchema]:
    logger.info(f"load dataframe from {file_path}")
    df: pd.DataFrame = pd.read_pickle(file_path)
    df = df.dropna()

    return df
