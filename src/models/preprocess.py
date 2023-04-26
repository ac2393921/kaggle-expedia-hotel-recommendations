from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA


class BasePreprocessPipeline(ABC):
    """前処理のベースクラス

    Args:
        ABC (_type_): _description_
    """

    def __init__(self):
        pass

    @abstractmethod
    def preprocess(
        self,
        x: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        pass


class DataPreprocessPipeline(BasePreprocessPipeline):
    """前処理を行うクラス

    Args:
        BasePreprocessPipeline (_type_): _description_
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return "DataPreprocessPipeline"

    def preprocess(
        self,
        x: pd.DataFrame,
        y=None,
    ) -> pd.DataFrame:
        """前処理を実行する

        Args:
            x (pd.DataFrame): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        # 年月日に変換
        x["date_time"] = pd.to_datetime(x["date_time"])
        x["year"] = x["date_time"].dt.year
        x["month"] = x["date_time"].dt.month

        # 予約日とチェックイン日の差分を計算
        x["srch_ci"] = pd.to_datetime(x["srch_ci"], format="%Y-%m-%d", errors="coerce")
        x["srch_co"] = pd.to_datetime(x["srch_co"], format="%Y-%m-%d", errors="coerce")

        x["date_time_month"] = x["date_time"].apply(lambda x: x.month)

        x["stay_dur"] = (x["srch_co"] - x["srch_ci"]).astype("timedelta64[D]")
        x["no_of_days_bet_booking"] = (x["srch_ci"] - x["date_time"]).astype(
            "timedelta64[D]"
        )

        # チェックイン日を年月日に変換
        x["Cin_day"] = x["srch_ci"].apply(lambda x: x.day)
        x["Cin_month"] = x["srch_ci"].apply(lambda x: x.month)
        x["Cin_year"] = x["srch_ci"].apply(lambda x: x.year)

        x["Cin_day"] = x["Cin_day"].fillna(26.0)
        x["Cin_month"] = x["Cin_month"].fillna(8.0)
        x["Cin_year"] = x["Cin_year"].fillna(2014.0)
        x["stay_dur"] = x["stay_dur"].fillna(1.0)
        x["no_of_days_bet_booking"] = x["no_of_days_bet_booking"].fillna(0.0)

        # 検索された距離をPCAで圧縮
        destinations = pd.read_csv("./project/data/destinations.csv")
        pca = PCA(n_components=3)
        dest_small = pca.fit_transform(
            destinations[["d{0}".format(i + 1) for i in range(149)]]
        )
        dest_small = pd.DataFrame(dest_small)
        dest_small["srch_destination_id"] = destinations["srch_destination_id"]

        props = {}
        for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
            props[prop] = getattr(x["date_time"].dt, prop)

        carryover = [
            p for p in x.columns if p not in ["date_time", "srch_ci", "srch_co"]
        ]
        for prop in carryover:
            props[prop] = x[prop]

        # 泊数を計算
        date_props = ["month", "day", "dayofweek", "quarter"]
        for prop in date_props:
            props["ci_{0}".format(prop)] = getattr(x["srch_ci"].dt, prop)
            props["co_{0}".format(prop)] = getattr(x["srch_co"].dt, prop)
        props["stay_span"] = (x["srch_co"] - x["srch_ci"]).astype("timedelta64[h]")

        ret = pd.DataFrame(props)

        ret = ret.join(dest_small, on="srch_destination_id", how="left", rsuffix="dest")
        ret = ret.drop("srch_destination_iddest", axis=1)

        # ダミー変数を作成
        # categorical_columns = [
        #     "hotel_continent",
        # ]
        # ret = pd.get_dummies(ret, columns=categorical_columns, sparse=True)

        ret.columns = ret.columns.astype(str)
        ret = ret.dropna()
        logger.info(ret.info())

        return ret
