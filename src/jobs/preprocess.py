import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.dataset.shema import TrainSchema, TestSchema, SplitData
from pandera.typing import DataFrame
import pandera as pa


@pa.check_types
def random_split(
    train_data: DataFrame[TrainSchema],
    target: DataFrame,
    test_size: float = 0.2,
) -> SplitData:
    logger.info("random split")

    x_train, x_test, y_train, y_test = train_test_split(
        train_data, target, test_size=test_size, shuffle=True
    )
    x_train, x_test, y_train, y_test = (
        TrainSchema(x_train),
        TestSchema(x_test),
        TestSchema(y_train),
        TestSchema(y_test),
    )

    return SplitData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
