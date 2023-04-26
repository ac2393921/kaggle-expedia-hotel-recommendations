from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from src.models.base_model import BaseRecommendModel
from src.models.eval import find_top_5, map_k


@dataclass
class Evaluation:
    mapk: float


class Artifact(BaseModel):
    preprocess_file_path: Optional[str]
    model_file_path: Optional[str]


class Trainer:
    def __init__(self) -> None:
        pass

    def train(
        self,
        model: BaseRecommendModel,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        logger.info("start train")
        model.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

    def evaluate(
        self,
        model: BaseRecommendModel,
        x: pd.DataFrame,
        y: pd.DataFrame,
    ) -> Evaluation:
        logger.info("start evaluation")
        predictions = model.predict(x=x)

        predictions_frame = pd.DataFrame(predictions)

        preds = []
        with tqdm(total=len(predictions_frame)) as pbar:
            for _, row in tqdm(predictions_frame.iterrows()):
                preds.append(find_top_5(row))
                pbar.update(1)

        mapk = map_k([[l] for l in y], preds, k=5)
        logger.info(f"MAP@5: {mapk}")

        evaluation = Evaluation(mapk)

        return evaluation

    def train_and_evaluate(
        self,
        model: BaseRecommendModel,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> Tuple[Evaluation, Artifact]:
        logger.info("start training and evaluation")
        self.train(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )

        evaluation = self.evaluate(
            model=model,
            x=x_test,
            y=y_test,
        )

        artifact = Artifact()

        logger.info("done training and evaluation")
        return evaluation, artifact
