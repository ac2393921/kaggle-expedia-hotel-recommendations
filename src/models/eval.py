import numpy as np


def map5eval(preds, dtrain, k=5):
    """MAP@5を計算する

    Args:
        preds (_type_): _description_
        dtrain (_type_): _description_
        k (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    actual = dtrain.get_label()
    predicted = (-preds).argsort(axis=1)[:, :k]
    metric = 0.0
    for i in range(5):
        metric += np.sum(actual == predicted[:, i]) / (i + 1)
    metric /= actual.shape[0]
    return "MAP@5", metric


def find_top_5(row):
    """トップ5のアイテムを取得する

    Args:
        row (_type_): _description_

    Returns:
        _type_: _description_
    """
    return list(row.nlargest(5).index)


def precision_k(actual: list, predicted: list, k: int = 10) -> float:
    """
    Precision@Kを測定するメソッド

    Parameters
    ----------
    actual : list
        実際にユーザが嗜好したアイテムのlist
    predicted : list
        レコメンドしたアイテムのlist
    k : int
        測定対象のアイテムtopN

    Returns
    ----------
    precision_k_value : float
        Precision@K
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])

    precision_k_value = len(actual_set & predicted_set) / k

    return precision_k_value


def ap_k(actual: list, predicted: list, k: int = 10) -> float:
    """
    AP@Kを測定するメソッド

    Parameters
    ----------
    actual : list
        実際にユーザが嗜好したアイテムのlist
    predicted : list
        レコメンドしたアイテムのlist
    k : int
        測定対象のアイテムtopN

    Returns
    ----------
    ap_k_value : float
        AP@K
    """
    ap_k_list = []
    for i in range(k):
        if len(actual) > i:
            ap_k_list.append(precision_k(actual, predicted, i + 1))

    ap_k_value = sum(ap_k_list) / min(len(actual), k)

    return ap_k_value


def map_k(actuals: list, predicteds: list, k: int = 10) -> list:
    """
    MAP@Kを測定するメソッド

    Parameters
    ----------
    actuals : list
        実際にユーザが嗜好したアイテムのlistのlist
    predicteds : list
        レコメンドしたアイテムのlistのlist
    k : int
        測定対象のアイテムtopN

    Returns
    ----------
    map_k_list : list
        MAP@K
    """
    map_k_list = []
    for i in range(k):
        ap_k_list = []
        for actual, predicted in zip(actuals, predicteds):
            ap_k_list.append(ap_k(actual, predicted, i + 1))
        map_k_list.append(sum(ap_k_list) / len(actuals))

    return np.mean(map_k_list)
