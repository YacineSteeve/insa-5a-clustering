from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict

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


MethodType = Literal["k_means", "agglo", "dbscan", "hdbscan"]
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
METHOD_TYPES: List[MethodType] = ["k_means", "agglo", "dbscan", "hdbscan"]


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
    arff_file = arff.loadarff(DATASET_DIR / file)

    data, _ = arff_file

    dataframe = pd.DataFrame(data)

    *other_columns, label_column = dataframe.columns
    features_columns = other_columns[:2]

    X = dataframe[features_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_categories = dataframe[label_column].astype("category").cat.codes.to_numpy()

    return X_scaled, y_categories, features_columns, label_column


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


def get_min_points_for_dbscan(X: npt.NDArray) -> int:
    return int(max(X.shape[1] + 1, np.log(X.shape[0])))


def process_for_kmeans(X: npt.NDArray, y: npt.NDArray, X_columns: List[str], y_column: str) -> None:
    k_clusters_silhouette = 0
    sil_score = 0
    n_clusters = list(range(2,30))
    for k in n_clusters:
        kmeans = KMeans(n_clusters=k).fit(X)
        labels = kmeans.labels_
        curr_score = silhouette_score(X, labels)
        if curr_score > sil_score:
            sil_score = curr_score
            k_clusters_silhouette = k

    best_kmeans = KMeans(n_clusters=k_clusters_silhouette, random_state=42).fit(X)
    labels = best_kmeans.labels_
    centroids = best_kmeans.cluster_centers_

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30, alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker=".", s=200, label="Centroids")
    plt.title(f"KMeans clustering avec k={k_clusters_silhouette} (silhouette score={sil_score:.2f})")
    plt.legend()
    plt.show()


def process_for_agglo(X: npt.NDArray, y: npt.NDArray, X_columns: List[str], y_column: str) -> None:
    k_clusters_silhouette = 0
    sil_score = 0
    n_clusters = list(range(2,30))
    for k in n_clusters:
        kmeans = AgglomerativeClustering(n_clusters=k).fit(X)
        labels = kmeans.labels_
        curr_score = silhouette_score(X, labels)
        if curr_score > sil_score:
            sil_score = curr_score
            k_clusters_silhouette = k

    best_kmeans = AgglomerativeClustering(n_clusters=k_clusters_silhouette).fit(X)
    labels = best_kmeans.labels_

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30, alpha=0.6)
    plt.title(f"Agglomerative Clustering avec k={k_clusters_silhouette} (silhouette score={sil_score:.2f})")
    plt.legend()
    plt.show()

def process_for_dbscan(X: npt.NDArray, y: npt.NDArray, X_columns: List[str], y_column: str) -> None:
    min_points = get_min_points_for_dbscan(X)

    # Compute k_distances
    nearest_neighbors = NearestNeighbors(n_neighbors=min_points)
    nearest_neighbors.fit(X)
    distances, _ = nearest_neighbors.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    # Find the elbow and the best epsilon
    n_k_distances = len(k_distances)
    knee_locator = KneeLocator(
        x=range(n_k_distances),
        y=k_distances,
        curve="convex",
        direction="increasing"
    )
    elbow = knee_locator.elbow
    epsilon0 = k_distances[elbow]
    window = n_k_distances // 20
    elbow_min, elbow_max = max(elbow - window, 0), min(elbow + window, n_k_distances - 1)
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
        silhouette_scores = np.append(silhouette_scores, silhouette_score(X, prediction))

    best_silhouette_score_index = np.argmax(silhouette_scores)
    best_silhouette_score = silhouette_scores[best_silhouette_score_index]
    best_epsilon = epsilons_range[best_silhouette_score_index]

    # Compute the best prediction
    best_dbscan = DBSCAN(eps=best_epsilon, min_samples=min_points)
    best_prediction = best_dbscan.fit_predict(X)

    # Plot results
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(k_distances)
    axes[0, 0].set_title("k_distances")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("k_distances")
    axes[0, 0].axhline(x=epsilon0, color="red")
    axes[0, 0].axhline(x=epsilon_min, color="green", linestyle="dashed")
    axes[0, 0].axhline(x=epsilon_max, color="green", linestyle="dashed")

    axes[0, 1].plot(epsilons_range, silhouette_scores)
    axes[0, 1].set_title("Silhouette score")
    axes[0, 1].set_xlabel("epsilon")
    axes[0, 1].set_ylabel("Silhouette score")
    axes[0, 1].axvline(x=best_epsilon, color="red", linestyle="dashed")
    axes[0, 1].axhline(y=best_silhouette_score, color="red")

    axes[1, 0].scatter(X[:, 0], X[:, 1], c=best_prediction)
    axes[1, 0].set_title("Clusters")
    axes[1, 0].set_xlabel(X_columns[0])
    axes[1, 0].set_ylabel(X_columns[1])

    axes[1, 1].scatter(X[:, 0], X[:, 1], c=y)
    axes[1, 1].set_title("Original clusters")
    axes[1, 1].set_xlabel(X_columns[0])
    axes[1, 1].set_ylabel(X_columns[1])

    plt.show()

def process_for_hdbscan(X: npt.NDArray, y: npt.NDArray, X_columns: List[str], y_column: str) -> None:
    min_points = get_min_points_for_dbscan(X)

    pass


def analyse_file(filename: str) -> None:
    X, y, X_columns, y_column = parse_file(filename)

    process_for_kmeans(X, y, X_columns, y_column)
    process_for_agglo(X, y, X_columns, y_column)
    process_for_dbscan(X, y, X_columns, y_column)
    process_for_hdbscan(X, y, X_columns, y_column)


if __name__ == "__main__":
    analyse_file("banana.arff")
