from argparse import ArgumentParser
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils.mdl_utils import (
    train_test_split, load_config,
    precision, recall, f1_score, fpr
)


class Classifier:
    """
    Classifier based on tf-idf
    """
    def __init__(self, data: pd.DataFrame) -> None:
        self.tv, self.test = train_test_split(data, 0.8)
        self.train, self.val = train_test_split(self.tv, 0.8)
        self.model = {
            "dt": DecisionTreeClassifier(),
            "rf": RandomForestClassifier(),
            "xgb": XGBClassifier()
        }
        self.eval_metric = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "fpr": fpr
        }


    def train_process(self, args: ArgumentParser) -> tuple:
        """
        train model
        """
        config = load_config(f"config/{args.model}_config.yaml")
        model = self.model[args.model](**config)
        model.fit(self.tv, self.tv["label"])
        prediction = model.predict(self.val)
        metric_dict = {}
        for key, val in self.eval_metric.items():
            metric_dict[key] = \
                self.eval_metric[val](prediction, self.val["label"])
        return model, metric_dict


    def test_process(self, model) -> None:
        """
        test
        """
        prediction = model.predict(self.test)
        return prediction
