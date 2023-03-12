from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier

from src.models.base_model import BaseRecommendModel
from sklearn.preprocessing import LabelEncoder

XGB_CLASSIFIER_DEFAULT_PARAMS = xgb_params = {
    # "tree_method": "gpu_hist",
    "objective": "multi:softmax",
    "num_class": 100,
    "learning_rate": 0.08,
    "max_depth": 8,
    "min_child_weight": 9,
    "silent": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "n_estimators": 600,
    "seed": 42,
}


class XGBClassifierRecommendModel(BaseRecommendModel):
    def __init__(self):
        self.name = "xgb"
        self.params: Dict = XGB_CLASSIFIER_DEFAULT_PARAMS
        self.model: XGBClassifier = None
        self.reset_model(params=self.params)

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

        self.model.fit(x_train, y_train, eval_set=eval_set, verbose=True)

    def predict(
        self, x: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        return super().predict(x)


if __name__ == "__main__":
    model = XGBClassifierRecommendModel()
