from loguru import logger

from src.jobs.retrieve import DataRetriever
from src.jobs.preprocess import random_split
from src.models.xgb_classifier import XGBClassifierRecommendModel


def main():
    logger.info("========データ取得========")
    data_retriever = DataRetriever()
    raw_data = data_retriever.retrieve_dataset()

    logger.info("========前処理========")
    split_data = random_split(**raw_data.dict())

    logger.info("========学習========")
    model = XGBClassifierRecommendModel()
    model.train(**split_data.dict())


if __name__ == "__main__":
    main()
