import os

import hydra
import mlflow
from loguru import logger
from omegaconf import DictConfig

from src.jobs.preprocess import random_split
from src.jobs.retrieve import DataRetriever
from src.jobs.train import Trainer
from src.models.xgb_classifier import XGBClassifierRecommendModel


@hydra.main(
    version_base=None,
    config_path="/src/conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    cwd = os.getcwd()
    run_name = "-".join(cwd.split("/")[-2:])
    logger.info(run_name)

    logger.info("========データ取得========")
    data_retriever = DataRetriever()
    raw_data = data_retriever.retrieve_dataset()

    logger.info("========前処理========")
    split_data = random_split(**raw_data.dict())

    mlflow.set_tracking_uri("./mlruns/")
    mlflow.set_experiment(cfg.name)
    logger.info(cfg.name)

    logger.info("========学習========")
    with mlflow.start_run(run_name=run_name):
        model = XGBClassifierRecommendModel()
        mlflow.log_param("model", model.name)

        trainer = Trainer()
        evaluation, artifact = trainer.train_and_evaluate(model, **split_data.dict())
        mlflow.log_metric("mep@5", evaluation.mapk)

if __name__ == "__main__":
    main()
