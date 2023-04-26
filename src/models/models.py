from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

from src.models.base_model import BaseRecommendModel
from src.models.randomforest_classifier import (
    RANDM_FOREST_CLASSIFIER_DEFAULT_PARAMS,
    RandomForestClassifierRecommendModel,
)
from src.models.xgb_classifier import (
    XGB_CLASSIFIER_DEFAULT_PARAMS,
    XGBClassifierRecommendModel,
)


class CustomModelConfig:
    arbitrary_types_allowed = True


class Model(BaseModel):
    name: str
    model: BaseRecommendModel
    params: Dict

    class Config(CustomModelConfig):
        pass


class MODELS(Enum):
    XGB_CLASSIFIER = Model(
        name="xgb_classifier",
        model=XGBClassifierRecommendModel(),
        params=XGB_CLASSIFIER_DEFAULT_PARAMS,
    )

    RANDM_FOREST_CLASSIFIER = Model(
        name="random_forest_classifier",
        model=RandomForestClassifierRecommendModel(),
        params=RANDM_FOREST_CLASSIFIER_DEFAULT_PARAMS,
    )

    @staticmethod
    def has_value(name: str) -> bool:
        return name in [v.value.name for v in MODELS.__members__.values()]

    @staticmethod
    def get_list() -> List[Model]:
        return [v.value for v in MODELS.__members__.values()]

    @staticmethod
    def get_model(name: str) -> Optional[Model]:
        for model in [v.value for v in MODELS.__members__.values()]:
            if model.name == name:
                return model
        raise ValueError
