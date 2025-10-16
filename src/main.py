from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from hdbscan import HDBSCAN
from kneed import KneeLocator
from scipy.io import arff
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# Type definitions
MethodType = Literal["k-means", "agglo", "dbscan", "hdbscan"]
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
Prediction = TypedDict(
    "Prediction",
    {
        "params": Params,
        "labels": npt.NDArray,
        "intermediary_plots_figures": List[plt.Figure]
    }
)


# Constants
DATASET_DIR = Path("dataset/artificial")
PLOTS_DIR = Path("plots")
ALL_METHODS_TYPES: List[MethodType] = ["k-means", "agglo", "dbscan", "hdbscan"]


def get_method_params_options(method_type: MethodType) -> ParamsOptions:
    match method_type:
        case "k-means":
            return [
                {
                    "n_clusters": n_clusters,
                    "init": init,
                    "n_init": n_init,
                    "max_iter": max_iter,
                    "tol": tol,
                    "algorithm": algorithm
                }
                for n_clusters in range(2, 20)
                for init in ["random", "k-means++"]
                for n_init in range(5, 25, 5)
                for max_iter in range(100, 600, 100)
                for tol in [10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
                for algorithm in ["lloyd", "elkan"]
            ]
        case "agglo":
            return [
                {
                    "n_clusters": n_clusters,
                    "distance_threshold": distance_threshold,
                    "metric": metric,
                    "linkage": linkage,
                }
                for n_clusters in range(2, 9)
                for distance_threshold in np.linspace(0, 5, 10)
                for metric in ["euclidean", "manhattan", "l1", "l2", "cosine"]
                for linkage in ["ward", "complete", "average", "single"]
            ]
        case "dbscan":
            return [
                {
                    "eps": eps,
                    "min_samples": min_samples,
                }
                for eps in np.linspace(0, 5, 10)
                for min_samples in range(np.log(20))
            ]
        case "hdbscan":
            return [

            ]


def parse_file(file: str) -> Tuple[npt.NDArray, npt.NDArray, List[str], str]:
    data, _ = arff.loadarff(DATASET_DIR / file)

    dataframe = pd.DataFrame(data)

    *other_columns, label_column = dataframe.columns
    features_columns = other_columns[:2]

    X = dataframe[features_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_categories = dataframe[label_column].astype("category").cat.codes.to_numpy()

    return X_scaled, y_categories, features_columns, label_column


def get_metrics_for_method(X: npt.NDArray, method: ClusterMixin) -> Metrics:
    prediction = method.fit_predict(X)

    return {
        "silhouette_score": silhouette_score(X, prediction),
        "calinski_harabasz_score": calinski_harabasz_score(X, prediction),
        "davies_bouldin_score": davies_bouldin_score(X, prediction)
    }


def plot_dataset(
    X_columns: List[str],
    X: npt.NDArray,
    y_column: str,
    y: npt.NDArray
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))

    first_column, second_column = X[:, 0], X[:, 1]

    axes[0].scatter(first_column, second_column)
    axes[0].set_title("Raw data")
    axes[0].set_xlabel(X_columns[0])
    axes[0].set_ylabel(X_columns[1])

    axes[1].scatter(first_column, second_column, c=y)
    axes[1].set_title(f"Defined clusters ({y_column})")
    axes[1].set_xlabel(X_columns[0])
    axes[1].set_ylabel(X_columns[1])

    figure.suptitle(f"Dataset")

    return figure


def plot_best_predictions(
    X_columns: List[str],
    X: npt.NDArray,
    best_predictions: Dict[MethodType, Prediction]
) -> plt.Figure:
    plots_counts = len(best_predictions)

    if plots_counts == 0:
        figure = plt.figure(figsize=(8, 6))
        figure.suptitle("Best predictions (no data)")

        return figure

    rows_count = plots_counts // 2 + plots_counts % 2
    figure, axes = plt.subplots(
        nrows=rows_count,
        ncols=2,
        figsize=(16, min(9 * rows_count, 50)),
        squeeze=False
    )

    first_column, second_column = X[:, 0], X[:, 1]

    for i, (method_type, prediction) in enumerate(best_predictions.items()):
        params = prediction["params"]
        labels = prediction["labels"]

        row, column = i // 2, i % 2
        axis = cast(plt.Axes, cast(object, axes[row, column]))

        axis.scatter(first_column, second_column, c=labels)
        axis.set_title(f"{method_type} ({params})")
        axis.set_xlabel(X_columns[0])
        axis.set_ylabel(X_columns[1])

    if plots_counts % 2 == 1:
        empty_axis = cast(plt.Axes, cast(object, axes[-1, -1]))
        empty_axis.axis("off")

    figure.suptitle(f"Best predictions")

    return figure


def get_min_points_for_dbscan(X: npt.NDArray) -> int:
    return int(max(X.shape[1] + 1, np.log(X.shape[0])))


def process_for_kmeans(X: npt.NDArray) -> Prediction:
    k_clusters_silhouette = 0
    sil_score = 0
    n_clusters = list(range(2,30))
    for k in n_clusters:
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(X)
        curr_score = silhouette_score(X, labels)
        if curr_score > sil_score:
            sil_score = curr_score
            k_clusters_silhouette = k

    best_kmeans = KMeans(n_clusters=k_clusters_silhouette, random_state=42)
    best_prediction = best_kmeans.fit_predict(X)
    centroids = best_kmeans.cluster_centers_

    figure, axes = plt.subplots(1, 1, figsize=(8, 6))
    axes.scatter(X[:, 0], X[:, 1], c=best_prediction, cmap="viridis", s=30, alpha=0.6)
    axes.scatter(centroids[:, 0], centroids[:, 1], c="red", marker=".", s=200, label="Centroids")
    axes.set_title(f"KMeans clustering avec k={k_clusters_silhouette} (silhouette score={sil_score:.2f})")

    return {
        "params": {
            "n_clusters": k_clusters_silhouette,
            "random_state": 42
        },
        "labels": best_prediction,
        "intermediary_plots_figures": [figure]
    }


def process_for_agglo(X: npt.NDArray) -> Prediction:
    k_clusters_silhouette = 0
    sil_score = 0
    n_clusters = list(range(2,30))
    for k in n_clusters:
        kmeans = AgglomerativeClustering(n_clusters=k)
        labels = kmeans.fit_predict(X)
        curr_score = silhouette_score(X, labels)
        if curr_score > sil_score:
            sil_score = curr_score
            k_clusters_silhouette = k

    best_agglo = AgglomerativeClustering(n_clusters=k_clusters_silhouette)
    best_prediction = best_agglo.fit_predict(X)

    figure, axes = plt.subplots(1, 1, figsize=(8, 6))
    axes.scatter(X[:, 0], X[:, 1], c=best_prediction, cmap="viridis", s=30, alpha=0.6)
    axes.set_title(f"Agglomerative clustering avec k={k_clusters_silhouette} (silhouette score={sil_score:.2f})")

    return {
        "params": {
            "n_clusters": k_clusters_silhouette,
        },
        "labels": best_prediction,
        "intermediary_plots_figures": [figure]
    }

def process_for_dbscan(X: npt.NDArray) -> Prediction:
    min_points = get_min_points_for_dbscan(X)

    # Compute k_distances
    nearest_neighbors = NearestNeighbors(n_neighbors=min_points)
    nearest_neighbors.fit(X)
    raw_k_distances, _ = nearest_neighbors.kneighbors()
    k_distances = np.sort(raw_k_distances[:, -1])

    # Find the elbow and the best epsilon
    k_distances_count = len(k_distances)
    knee_locator = KneeLocator(
        x=range(k_distances_count),
        y=k_distances,
        curve="convex",
        direction="increasing"
    )
    elbow = knee_locator.elbow
    epsilon0 = k_distances[elbow]
    window = k_distances_count // 20
    elbow_min, elbow_max = max(elbow - window, 0), min(elbow + window, k_distances_count - 1)
    left_slope = (k_distances[elbow] - k_distances[elbow_min]) / (elbow - elbow_min + 1e-9)
    right_slope = (k_distances[elbow_max] - k_distances[elbow]) / (elbow_max - elbow + 1e-9)
    sharpness = right_slope / (left_slope + 1e-9)
    sharpness = np.clip(sharpness, 1, 10)
    pct = np.interp(sharpness, [1, 10], [0.3, 0.05])
    epsilon_min = epsilon0 * (1 - pct)
    epsilon_max = epsilon0 * (1 + pct)

    # Find the best epsilon
    epsilons_range = np.linspace(epsilon_min, epsilon_max, 10)
    silhouette_scores = np.array([])

    for epsilon in epsilons_range:
        dbscan = DBSCAN(eps=epsilon, min_samples=min_points)
        prediction = dbscan.fit_predict(X)
        if len(set(prediction)) == 1:
            silhouette_scores = np.append(silhouette_scores, 0)
        else:
            silhouette_scores = np.append(silhouette_scores, silhouette_score(X, prediction))

    best_silhouette_score_index = np.argmax(silhouette_scores)
    best_silhouette_score = silhouette_scores[best_silhouette_score_index]
    best_epsilon = epsilons_range[best_silhouette_score_index]

    # Compute the best prediction
    best_dbscan = DBSCAN(eps=best_epsilon, min_samples=min_points)
    best_prediction = best_dbscan.fit_predict(X)

    # Plot results
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(k_distances)
    axes[0].set_title("k_distances")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("k_distances")
    axes[0].axhline(y=epsilon0, color="red")
    axes[0].axhline(y=epsilon_min, color="green", linestyle="dashed")
    axes[0].axhline(y=epsilon_max, color="green", linestyle="dashed")

    axes[1].plot(epsilons_range, silhouette_scores)
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("Silhouette score")
    axes[1].axvline(x=best_epsilon, color="red", linestyle="dashed")
    axes[1].axhline(y=best_silhouette_score, color="red")

    return {
        "params": {
            "eps": float(best_epsilon),
            "min_samples": min_points,
        },
        "labels": best_prediction,
        "intermediary_plots_figures": [figure]
    }


def process_for_hdbscan(X: npt.NDArray) -> Prediction:
    return {
        "params": {},
        "labels": HDBSCAN().fit_predict(X),
        "intermediary_plots_figures": []
    }


def process_for_method(method_type: MethodType, X: npt.NDArray) -> Prediction:
    match method_type:
        case "k-means":
            return process_for_kmeans(X)
        case "agglo":
            return process_for_agglo(X)
        case "dbscan":
            return process_for_dbscan(X)
        case "hdbscan":
            return process_for_hdbscan(X)
        case _:
            raise ValueError(f"Unknown method type: {method_type}")


def analyse_file(filename: str, method_types: Optional[List[MethodType]]) -> None:
    if method_types is None:
        method_types = ALL_METHODS_TYPES

    X, y, X_columns, y_column = parse_file(filename)

    best_predictions = {
        method_type: process_for_method(method_type, X)
        for method_type in method_types
    }

    PLOTS_DIR.mkdir(exist_ok=True)

    directory = PLOTS_DIR / filename.split(".")[0]
    directory.mkdir(exist_ok=True)

    dataset_figure = plot_dataset(X_columns, X, y_column, y)
    dataset_figure.tight_layout()
    dataset_figure.savefig(directory / "dataset.png")
    plt.close(dataset_figure)

    best_predictions_figure = plot_best_predictions(X_columns, X, best_predictions)
    best_predictions_figure.tight_layout()
    best_predictions_figure.savefig(directory / f"best_predictions_{'_'.join(sorted(method_types))}.png")
    plt.close(best_predictions_figure)

    for method_type, prediction in best_predictions.items():
        figures_count = len(prediction["intermediary_plots_figures"])
        for i in range(figures_count):
            intermediary_plots_figure = prediction["intermediary_plots_figures"][i]
            intermediary_plots_figure.tight_layout()
            intermediary_plots_figure.savefig(directory / f"intermediary_plots_{method_type}{f'_{i+1}' if figures_count > 1 else ''}.png")
            plt.close(intermediary_plots_figure)


if __name__ == "__main__":
    for arff_file in ["diamond9.arff", "banana.arff"]:
        analyse_file(arff_file, ["k-means", "agglo"])

    for arff_file in ["diamond9.arff", "banana.arff"]:
        analyse_file(arff_file, ["dbscan", "hdbscan"])
