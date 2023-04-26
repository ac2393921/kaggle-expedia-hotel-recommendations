from loguru import logger

from src.dataset.data_manager import load_df_from_csv, load_test_df_from_csv
from src.dataset.shema import (
    TARGET_SCHEMA,
    HotelTestSchema,
    HotelTrainSchema,
    RawData,
    TestData,
)


class DataRetriever:
    def __init__(self) -> None:
        pass

    def retrieve_dataset(self, path: str, nrows: int = 10000) -> RawData:
        logger.info("start retieve data")
        raw_df = load_df_from_csv(path, nrows)
        train_df = HotelTrainSchema(raw_df)
        target = raw_df.hotel_cluster
        logger.info(target)
        TARGET_SCHEMA.validate(target)
        raw_data = RawData(train_data=train_df, target=target)

        return raw_data

    def retrieve_test_dataset(self, path: str) -> TestData:
        logger.info("start retieve data")
        raw_df = load_test_df_from_csv(path)
        raw_id = raw_df["id"]
        logger.info(raw_df)
        logger.info(raw_df.info())
        raw_df = HotelTestSchema(raw_df)

        # raw_df["date_time"] = pd.to_datetime(raw_df["date_time"])
        # raw_df["year"] = raw_df["date_time"].dt.year
        # raw_df["month"] = raw_df["date_time"].dt.month

        logger.info(raw_df.columns)
        logger.info(raw_df.info())
        # train_df = TrainSchema(train_df)
        # target = raw_df.hotel_cluster
        # TARGET_SCHEMA.validate(target)
        raw_data = TestData(id=raw_id, x=raw_df)

        return raw_data
