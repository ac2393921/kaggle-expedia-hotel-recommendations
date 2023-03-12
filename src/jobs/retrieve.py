from loguru import logger

from src.dataset.data_manager import load_df_from_csv
from src.dataset.shema import BaseSchema, RawData, TrainSchema, TARGET_SCHEMA


class DataRetriever:
    def __init__(self) -> None:
        pass

    def retrieve_dataset(self) -> RawData:
        logger.info("start retieve data")
        raw_df = load_df_from_csv("../project/data/train.csv", 10000)
        raw_df = BaseSchema(raw_df)

        train_df = raw_df.drop(
            [
                "hotel_cluster",
                "date_time",
                "srch_ci",
                "srch_co",
                "is_booking",
                "cnt",
            ],
            axis=1,
        )
        train_df = TrainSchema(train_df)
        target = raw_df.hotel_cluster
        TARGET_SCHEMA.validate(target)
        raw_data = RawData(train_data=train_df, target=target)

        return raw_data
