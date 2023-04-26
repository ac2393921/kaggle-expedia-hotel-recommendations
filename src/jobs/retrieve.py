from loguru import logger

from src.dataset.data_manager import load_df_from_csv, load_test_df_from_csv
from src.dataset.shema import (TARGET_SCHEMA, HotelTestSchema,
                               HotelTrainSchema, RawData, TestData)


class DataRetriever:
    """データを取得するクラス"""

    def __init__(self) -> None:
        pass

    def retrieve_dataset(self, path: str, nrows: int = 10000) -> RawData:
        """データを取得する

        Args:
            path (str): _description_
            nrows (int, optional): _description_. Defaults to 10000.

        Returns:
            RawData: _description_
        """
        logger.info("start retieve data")
        raw_df = load_df_from_csv(path, nrows)
        train_df = HotelTrainSchema(raw_df)
        target = raw_df.hotel_cluster
        logger.info(target)
        TARGET_SCHEMA.validate(target)
        raw_data = RawData(train_data=train_df, target=target)

        return raw_data

    def retrieve_test_dataset(self, path: str) -> TestData:
        """テストデータを取得する"""
        logger.info("start retieve data")
        raw_df = load_test_df_from_csv(path)
        raw_id = raw_df["id"]
        logger.info(raw_df)
        logger.info(raw_df.info())
        raw_df = HotelTestSchema(raw_df)

        logger.info(raw_df.columns)
        logger.info(raw_df.info())
        raw_data = TestData(id=raw_id, x=raw_df)

        return raw_data
