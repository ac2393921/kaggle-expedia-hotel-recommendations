import pandas as pd
from loguru import logger

from src.dataset.data_manager import load_df_from_csv


class DataRetriever:
    def __init__(self) -> None:
        pass

    def retrieve_dataset(self) -> pd.DataFrame:
        logger.info("start retieve data")
        raw_df = load_df_from_csv("../project/data/train.csv", 1000)

        return raw_df
