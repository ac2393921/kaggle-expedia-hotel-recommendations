from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.base_model import BaseRecommendModel

XGB_CLASSIFIER_DEFAULT_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.01,
    "n_estimators": 50,
    "random_state": 3,
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
        self.models = XGBClassifier(**self.params)

    def train(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        x_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):
        eval_set = [(x_train, y_train)]
        if x_test is not None and y_test is not None:
            eval_set.append((x_test, y_test))
        self.model.fit(x_train, y_train, eval_set=eval_set, verbose=True)


if __name__ == "__main__":
    model = XGBClassifier()
