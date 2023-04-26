from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class BaseRecommendModel(ABC):
    """レコメンドモデルのベースクラス"""

    def __init__(self):
        self.name: str = "base_beverage_hotel_recommend"
        self.params: Dict = {}
        self.model = None

    @abstractmethod
    def reset_model(self, params: Optional[Dict] = None) -> None:
        """モデルのパラメータをリセットする

        Args:
            params (Optional[Dict], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        x_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ):
        """モデルの学習を行う

        Args:
            x_train (Union[np.ndarray, pd.DataFrame]): _description_
            y_train (Union[np.ndarray, pd.DataFrame]): _description_
            x_test (Optional[Union[np.ndarray, pd.DataFrame]], optional): _description_. Defaults to None.
            y_test (Optional[Union[np.ndarray, pd.DataFrame]], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, x: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """モデルの予測を行う

        Args:
            x (Union[np.ndarray, pd.DataFrame]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Union[np.ndarray, pd.DataFrame]: _description_
        """
        raise NotImplementedError
