import pandera as pa
from loguru import logger
from pandera.typing import DataFrame
from sklearn.model_selection import train_test_split

from src.dataset.shema import SplitData, TrainSchema
from src.models.preprocess import BasePreprocessPipeline


@pa.check_types
def random_split(
    train_data: DataFrame[TrainSchema],
    target: DataFrame,
    test_size: float = 0.2,
) -> SplitData:
    """ランダムにデータを分割する

    Args:
        train_data (DataFrame[TrainSchema]): _description_
        target (DataFrame): _description_
        test_size (float, optional): _description_. Defaults to 0.2.

    Returns:
        SplitData: _description_
    """
    logger.info("random split")

    x_train, x_test, y_train, y_test = train_test_split(
        train_data, target, test_size=test_size, shuffle=True
    )

    x_train, x_test, y_train, y_test = (
        TrainSchema(x_train),
        TrainSchema(x_test),
        y_train,
        y_test,
    )

    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def train_test_split(data: any, data_preprocess_pipeline: BasePreprocessPipeline):
    """訓練データとテストデータに分割する。テストデータは2014年8月以降のデータのみを使用する
        前処理パイプラインを設定することで、前処理済みのデータを返す

    Args:
        data (any): _description_
        data_preprocess_pipeline (BasePreprocessPipeline): _description_

    Returns:
        _type_: _description_
    """
    preprocess_data = data_preprocess_pipeline.preprocess(data)

    # 2013年と2014年8月以前のデータを訓練データとする
    train = preprocess_data[
        (
            (preprocess_data.year == 2013)
            | ((preprocess_data.year == 2014) & (preprocess_data.month < 8))
        )
    ]
    test = preprocess_data[
        ((preprocess_data.year == 2014) & (preprocess_data.month >= 8))
    ]
    test = test[test.is_booking == 1]

    y_train = train.hotel_cluster
    y_test = test.hotel_cluster

    x_train = train.drop(
        [
            "hotel_cluster",
            "is_booking",
            "cnt",
        ],
        axis=1,
    )
    x_test = test.drop(
        [
            "hotel_cluster",
            "is_booking",
            "cnt",
        ],
        axis=1,
    )

    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
