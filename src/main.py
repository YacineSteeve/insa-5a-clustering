from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from scipy.io import arff
from sklearn.base import ClusterMixin
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

DATASET_DIR = Path("dataset/artificial")
FEATURES_COLUMS = ["a0", "a1"]
LABEL_COLUMN = "class"

Metrics = TypedDict(
    "Metrics", 
    {
        "silhouette_score": float, 
        "calinski_harabasz_score": float, 
        "davies_bouldin_score": float
    }
)
Params = Dict[str, Any]
ParamsOptions = List[Params]

def get_k_means_params() -> ParamsOptions:
    return [
        {
            "n_clusters": a,
            "init": b,
            "n_init": c,
            "max_iter": d,
            "tol": 1/e,
            "algorithm": f
        }
        for a in range(2, 9)
        for b in ["random", "k-means++"]
        for c in range(5, 50, 5)
        for d in range(100, 1100, 100)
        for e in [10^2, 10^3, 10^4, 10^5, 10^6]
        for f in ["lloyd", "elkan"]
    ]

def get_dbscan_params() -> ParamsOptions:
    return [

    ]

def get_hdbscan_params() -> ParamsOptions:
    return [

    ]

def get_hierarch_params() -> ParamsOptions:
    return [

    ]

def parse_file(file: str) -> Tuple[Any, Any]:
    arff_file = arff.loadarff(DATASET_DIR / file)

    data, _ = arff_file

    dataframe = pd.DataFrame(data)

    return dataframe[FEATURES_COLUMS], dataframe[[LABEL_COLUMN]]

def get_metrics_for_method(
        X: Any,
        y: Any,
        method: ClusterMixin
    ) -> Metrics:
    prediction = method.fit_predict(X)

    return {
        "silhouette_score": silhouette_score(X, prediction),
        "calinski_harabasz_score": calinski_harabasz_score(X, prediction),
        "davies_bouldin_score": davies_bouldin_score(X, prediction)
    }


def analyse_file(file: str) -> None:
    X, y = parse_file(file)

    for method_type, params_options in [
        ("k_means", get_k_means_params()),
        ("dbscan", get_dbscan_params()),
        ("hdbscan", get_hdbscan_params()),
        ("hierarch", get_hierarch_params()),
    ]:
        for params in params_options:
            method: ClusterMixin

            if method_type == "k_means":
                method = KMeans(**params)
            elif method_type == "dbscan":
                method = DBSCAN(**params)
            elif method_type == "hdbscan":
                method = HDBSCAN(**params)
            elif method_type == "hierarch":
                method = AgglomerativeClustering(**params)

            metrics = get_metrics_for_method(
                X=X,
                y=y,
                method=method
            )

            print(f'Silhouette score: {metrics["silhouette_score"]}')
            print(f'Calinski-Harabasz score: {metrics["calinski_harabasz_score"]}')
            print(f'Davies-Bouldin score: {metrics["davies_bouldin_score"]}')


if __name__ == "__main__":
    analyse_file("2d-3c-no123.arff")
