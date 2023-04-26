from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src.models.base_model import BaseRecommendModel
from src.models.eval import map5eval

XGB_CLASSIFIER_DEFAULT_PARAMS = {}


class XGBClassifierRecommendModel(BaseRecommendModel):
    """XGBoostを使用したレコメンドモデル

    Args:
        BaseRecommendModel (_type_): _description_
    """

    def __init__(self):
        self.name = "XGBoost Classifier"
        self.params: Dict = XGB_CLASSIFIER_DEFAULT_PARAMS
        self.model: XGBClassifier = None
        self.reset_model(params=self.params)

        self.pipe = Pipeline(
            steps=[
                ("standerd_scaler", StandardScaler()),
                ("clf", self.model),
            ]
        )

    def reset_model(self, params: Optional[Dict] = None) -> None:
        if params is not None:
            self.params = params
        logger.info(f"params: {self.params}")
        self.model = XGBClassifier(**self.params)
        logger.info(f"initialized model: {self.model}")

    def train(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        x_test: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

        eval_set = [(x_test, y_test)]
        if x_test is not None and y_test is not None:
            eval_set.append((x_test, y_test))

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

        self.model.fit(
            x_train,
            y_train,
            eval_set=eval_set,
            verbose=True,
            eval_metric=map5eval,
        )

    def predict(
        self, x: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        return self.model.predict_proba(x)


if __name__ == "__main__":
    model = XGBClassifierRecommendModel()
