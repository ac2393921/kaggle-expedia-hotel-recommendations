from loguru import logger

from src.jobs.retrieve import DataRetriever
from src.jobs.preprocess import random_split


def main():
    logger.info("========データ取得========")
    data_retriever = DataRetriever()
    raw_data = data_retriever.retrieve_dataset()

    logger.info("========前処理========")
    split_data = random_split(
        train_data=raw_data.train_data,
        target=raw_data.target,
    )

    logger.info(split_data)


if __name__ == "__main__":
    main()
