from loguru import logger

from src.jobs.retrieve import DataRetriever


def main():
    data_retriever = DataRetriever()
    raw_data = data_retriever.retrieve_dataset()
    logger.info(raw_data.info())


if __name__ == "__main__":
    main()
