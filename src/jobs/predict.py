import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.models.base_model import BaseRecommendModel
from src.models.eval import find_top_5
from src.models.preprocess import BasePreprocessPipeline

from tying import List


class Predicter:
    """評価用のクラス"""

    def __init__(self) -> None:
        pass

    def predict(
        self,
        model: BaseRecommendModel,
        x,
        data_preprocess_pipeline: BasePreprocessPipeline,
    ) -> List[float]:
        """予測を行う

        Args:
            model (BaseRecommendModel): _description_
            x (_type_): _description_
            data_preprocess_pipeline (BasePreprocessPipeline): _description_

        Returns:
            List[float]: _description_
        """
        logger.info("start predict")
        x = data_preprocess_pipeline.preprocess(x)
        x = x.drop(
            [
                "id",
            ],
            axis=1,
        )
        predictions = model.predict(x=x)

        predictions_frame = pd.DataFrame(predictions)

        preds = []
        with tqdm(total=len(predictions_frame)) as pbar:
            for _, row in tqdm(predictions_frame.iterrows()):
                preds.append(" ".join(list(map(str, find_top_5(row)))))
                pbar.update(1)

        return preds
