import yaml
import numpy as np
import pandas as pd


def load_config(path: str) -> dict:
    """
    load config file
    """
    with open(path, "r", encoding="utf-8") as yml_file:
        config = yaml.safe_load(yml_file)
    return config


def train_test_split(data: pd.DataFrame, split_size: float) -> tuple:
    """
    split data into train and test set
    """
    n = len(data)
    tr_size = int(n * split_size)
    tr_idx = np.random.choice(data.index, tr_size, replace=False)
    ts_idx = np.setdiff1d(data.index, tr_idx)
    return data.loc[tr_idx], data.loc[ts_idx]


def precision(pred: pd.Series, true: pd.Series) -> float:
    """
    precision
    """
    return (pred & true).sum() / pred.sum()


def recall(pred: pd.Series, true: pd.Series) -> float:
    """
    recall
    """
    return (pred & true).sum() / true.sum()


def f1_score(pred: pd.Series, true: pd.Series) -> float:
    """
    f1 score
    """
    p_score = precision(pred, true)
    r_score = recall(pred, true)
    return (2 * p_score * r_score) / (p_score + r_score)


def fpr(pred: pd.Series, true: pd.Series) -> float:
    """
    false positive rate
    """
    return (pred & ~true).sum() / (~true).sum()
