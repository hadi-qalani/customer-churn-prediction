from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

MODELS = {"LogisticRegression": LogisticRegression}


def create_model(name: str, params: dict) -> BaseEstimator:

    return MODELS[name](**params)
