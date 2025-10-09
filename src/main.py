from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict

import numpy.typing as npt
import pandas as pd
from hdbscan import HDBSCAN
from scipy.io import arff
from sklearn.base import ClusterMixin
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


MethodType = Literal["k_means", "dbscan", "hdbscan", "hierarch"]
Params = Dict[str, Any]
ParamsOptions = List[Params]
Metrics = TypedDict(
    "Metrics", 
    {
        "silhouette_score": float, 
        "calinski_harabasz_score": float, 
        "davies_bouldin_score": float
    }
)


DATASET_DIR = Path("dataset/artificial")
FEATURES_COLUMS = ["a0", "a1"]
LABEL_COLUMN = "class"
METHOD_TYPES: List[MethodType] = ["k_means", "dbscan", "hdbscan", "hierarch"]


def get_method_params_options(method_type: MethodType) -> ParamsOptions:
    match method_type:
        case "k_means":
            return [
                {
                    "n_clusters": n_clusters,
                    "init": init,
                    "n_init": n_init,
                    "max_iter": max_iter,
                    "tol": tol,
                    "algorithm": algorithm
                }
                for n_clusters in range(2, 9)
                for init in ["random", "k-means++"]
                for n_init in range(5, 25, 5)
                for max_iter in range(100, 600, 100)
                for tol in [10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
                for algorithm in ["lloyd", "elkan"]
            ]
        case "dbscan":
            return [

            ]
        case "hdbscan":
            return [

            ]
        case "hierarch":
            return [

            ]
        case _:
            return []


def parse_file(file: str) -> Tuple[npt.NDArray, npt.NDArray]:
    arff_file = arff.loadarff(DATASET_DIR / file)

    data, _ = arff_file

    dataframe = pd.DataFrame(data)

    return dataframe[FEATURES_COLUMS].to_numpy(), dataframe[[LABEL_COLUMN]].to_numpy()


def get_metrics_for_method(
        X: npt.NDArray,
        y: npt.NDArray,
        method: ClusterMixin
    ) -> Metrics:
    prediction = method.fit_predict(X)

    return {
        "silhouette_score": silhouette_score(X, prediction),
        "calinski_harabasz_score": calinski_harabasz_score(X, prediction),
        "davies_bouldin_score": davies_bouldin_score(X, prediction)
    }


def analyse_file(filename: str) -> None:
    X, y = parse_file(filename)

    for method_type in METHOD_TYPES:
        params_options = get_method_params_options(method_type=method_type)
        params_options_count = len(params_options)

        if params_options_count  == 0:
            print(f"No params options to test for method '{method_type}', skipping this method")
            continue

        print(f"Testing {params_options_count} params options for method '{method_type}'...")

        for params in params_options:
            method: ClusterMixin

            match method_type:
                case "k_means":
                    method = KMeans(**params)
                case "dbscan":
                    method = DBSCAN(**params)
                case "hdbscan":
                    method = HDBSCAN(**params)
                case "hierarch":
                    method = AgglomerativeClustering(**params)
                case _:
                    method = KMeans(**params)

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
