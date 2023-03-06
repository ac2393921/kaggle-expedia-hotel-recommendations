import pandas as pd
from loguru import logger

from src.dataset.data_manager import load_df_from_csv
from src.dataset.shema import BaseSchema, RawData, TrainSchema, TestSchema


class DataRetriever:
    def __init__(self) -> None:
        pass

    def retrieve_dataset(self) -> RawData:
        logger.info("start retieve data")
        raw_df = load_df_from_csv("../project/data/train.csv", 1000)
        raw_df = BaseSchema(raw_df)

        train_df = TrainSchema(raw_df)
        target = TestSchema(raw_df)
        raw_data = RawData(train_data=train_df, target=target)

        return raw_data
