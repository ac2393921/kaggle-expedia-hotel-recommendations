import os

import hydra
import mlflow
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from src.jobs.predict import Predicter
from src.jobs.preprocess import train_test_split
from src.jobs.retrieve import DataRetriever
from src.jobs.train import Trainer
from src.models.models import MODELS
from src.models.preprocess import DataPreprocessPipeline


@hydra.main(
    version_base=None,
    config_path="/src/conf",
    config_name="xgb",
)
def main(cfg: DictConfig) -> None:
    cwd = os.getcwd()
    run_name = "-".join(cwd.split("/")[-2:])
    logger.info(run_name)

    logger.info("========データ取得========")
    data_retriever = DataRetriever()
    raw_data = data_retriever.retrieve_dataset(cfg.jobs.data.train.path, 100000)
    test_data = data_retriever.retrieve_test_dataset(cfg.jobs.data.test.path)

    mlflow.set_tracking_uri("./mlruns/")
    mlflow.set_experiment(cfg.name)
    logger.info(cfg.name)

    with mlflow.start_run(run_name=run_name):
        logger.info("========前処理========")
        data_preprocess_pipeline = DataPreprocessPipeline()
        split_data = train_test_split(
            raw_data.train_data,
            data_preprocess_pipeline,
        )
        mlflow.log_param("preprocess", data_preprocess_pipeline)

        logger.info("========学習========")
        _model = MODELS.get_model(name=cfg.jobs.model.name)
        model = _model.model
        if "params" in cfg.jobs.model.keys():
            model.reset_model(params=cfg.jobs.model.params)
        mlflow.log_param("model", model.name)

        trainer = Trainer()
        evaluation, _ = trainer.train_and_evaluate(
            model,
            **split_data.dict(),
        )
        mlflow.log_metric("map5", evaluation.mapk)

        # predict
        predicter = Predicter()
        test_pred = predicter.predict(
            model=model,
            x=test_data.x,
            data_preprocess_pipeline=data_preprocess_pipeline,
        )

    test_pred = pd.Series(test_pred)
    submission = pd.DataFrame()
    submission["id"] = test_data.id
    submission["hotel_cluster"] = test_pred
    submission.to_csv(f"submission/{model.name}.csv", index=False)


if __name__ == "__main__":
    main()
