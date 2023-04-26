from typing import Optional

import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import DataFrame

from src.dataset.shema import DTYPES, HotelTrainBaseSchema


@pa.check_types
def load_df_from_csv(
    file_path: str, nrows: Optional[int] = None
) -> DataFrame[HotelTrainBaseSchema]:
    """csvファイルを読み込み、pandas.DataFrameに変換する

    Args:
        file_path (str): _description_
        nrows (Optional[int], optional): _description_. Defaults to None.

    Returns:
        DataFrame[HotelTrainBaseSchema]: _description_
    """
    logger.info(f"load dataframe from {file_path}")
    df = pd.read_csv(file_path, nrows=nrows, dtype=DTYPES)
    df = df.dropna()

    return df


def load_test_df_from_csv(file_path: str, nrows: Optional[int] = None):
    """テストのcsvファイルを読み込み、pandas.DataFrameに変換する

    Args:
        file_path (str): _description_
        nrows (Optional[int], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    logger.info(f"load dataframe from {file_path}")
    df = pd.read_csv(file_path, nrows=nrows, dtype=DTYPES)
    logger.info(df.shape)

    return df


@pa.check_types
def load_df_from_pickle(file_path: str) -> DataFrame[HotelTrainBaseSchema]:
    """pickleファイルを読み込み、pandas.DataFrameに変換する

    Args:
        file_path (str): _description_

    Returns:
        DataFrame[HotelTrainBaseSchema]: _description_
    """
    logger.info(f"load dataframe from {file_path}")
    df: pd.DataFrame = pd.read_pickle(file_path)
    df = df.dropna()

    return df
