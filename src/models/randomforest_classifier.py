from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseRecommendModel

RANDM_FOREST_CLASSIFIER_DEFAULT_PARAMS = {}


class RandomForestClassifierRecommendModel(BaseRecommendModel):
    def __init__(self):
        self.name = "Random Forest Classifier"
        self.params: Dict = RANDM_FOREST_CLASSIFIER_DEFAULT_PARAMS
        self.model: RandomForestClassifier = None
        self.reset_model(params=self.params)

    def reset_model(self, params: Optional[Dict] = None) -> None:
        if params is not None:
            self.params = params
        logger.info(f"params: {self.params}")
        self.model = RandomForestClassifier(**self.params)
        logger.info(f"initialized model: {self.model}")

    def train(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        x_test: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):
        x_train = np.nan_to_num(x_train)
        y_train = np.nan_to_num(y_train)
        self.model.fit(x_train, y_train)

    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        x = np.nan_to_num(x)
        return self.model.predict_proba(x)
