from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
from scipy.io import arff
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid


warnings.filterwarnings("ignore")


# Type definitions
MethodType = Literal["k-means", "agglo", "dbscan", "hdbscan"]
Scorer = Literal["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
Prediction = TypedDict(
    "Prediction",
    {
        "params": Dict[str, Any],
        "labels": npt.NDArray,
        "score": float,
        "additional_plotter": Callable[[plt.Axes], None]
    }
)
ExperimentResult = Dict[Scorer, Prediction]


# Constants
DATASET_DIR = Path("dataset/artificial")
PLOTS_DIR = Path("plots/absolute2")
SCORERS: Dict[Scorer, Callable[[npt.NDArray, npt.NDArray], float]] = {
    "silhouette_score": silhouette_score,
    "calinski_harabasz_score": calinski_harabasz_score,
    "davies_bouldin_score": davies_bouldin_score
}
SCORER_COMPARATORS: Dict[Scorer, Any] = {
    "silhouette_score": max,
    "calinski_harabasz_score": max,
    "davies_bouldin_score": min
}
EPSILON_RANGE_PER_FILE: Dict[str, Tuple[float, float]] = {
    "atom": (0.075, 0.15),
    "banana": (0.05, 0.1),
    "diamond9": (0.09, 0.13),
    "jain": (0.17, 0.35),
    "spiral": (0.1, 0.14),
    "twenty": (0.1, 0.17),
}


def parse_file(file: str) -> Tuple[npt.NDArray, npt.NDArray, List[str], str]:
    data, _ = arff.loadarff(DATASET_DIR / file)

    dataframe = pd.DataFrame(data)

    *other_columns, label_column = dataframe.columns
    features_columns = other_columns[:2]

    X = dataframe[features_columns]
    y_categories = dataframe[label_column].astype("category").cat.codes.to_numpy()

    return X.to_numpy(), y_categories, features_columns, label_column


def plot_result(
    method_type: MethodType,
    X_columns: List[str],
    X: npt.NDArray,
    y_column: str,
    y: npt.NDArray,
    prediction: Prediction,
    scorer: Scorer,
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 12))

    first_column, second_column = X[:, 0], X[:, 1]

    axes[0][0].scatter(first_column, second_column, c=y)
    axes[0][0].set_title(f"Expected clusters {y_column}")
    axes[0][0].set_xlabel(X_columns[0])
    axes[0][0].set_ylabel(X_columns[1])

    axes[0][1].scatter(first_column, second_column, c=prediction["labels"])
    axes[0][1].set_title(f"Predicted clusters")
    axes[0][1].set_xlabel(X_columns[0])
    axes[0][1].set_ylabel(X_columns[1])

    prediction["additional_plotter"](axes[1][0])

    axes[1][1].axis("off")

    infos = {
        **prediction["params"],
        scorer: prediction["score"]
    }

    figure.suptitle(f"Result of {method_type} ({",".join(f"{key}: {value}" for key, value in infos.items())})")

    return figure


def blank_plotter(figure: plt.Figure) -> None:
    pass


def process_for_kmeans(X: npt.NDArray) -> ExperimentResult:
    params_grid = {
        "n_clusters": range(2, 30),
        "init": ["random", "k-means++"],
    }

    results = defaultdict(list)
    for params in ParameterGrid(params_grid):
        model = KMeans(**params)
        predicted_labels = model.fit_predict(X)
        if len(set(predicted_labels)) < 2:
            continue
        for scorer_name, scorer in SCORERS.items():
            score = scorer(X, predicted_labels)
            results[scorer_name].append([score, params, predicted_labels])

    experiment_results: ExperimentResult = {}
    for scorer_name, scorer_results in results.items():
        comparator = SCORER_COMPARATORS[scorer_name]
        best_result = comparator(scorer_results, key=lambda x: x[0])

        experiment_results[scorer_name] = {
            "score": best_result[0],
            "params": best_result[1],
            "labels": best_result[2],
            "additional_plotter": blank_plotter,
        }

        if scorer_name == "silhouette_score":
            def plot_metadata(figure: plt.Axes) -> None:
                silhouette_score_results = filter(
                    lambda result: result[1]["init"] == best_result[1]["init"],
                    results["silhouette_score"]
                )
                sorted_results = sorted(silhouette_score_results, key=lambda x: x[1]["n_clusters"])
                n_clusters = [result[1]["n_clusters"] for result in sorted_results]
                scores = [result[0] for result in sorted_results]
                figure.plot(n_clusters, scores, marker="o")
                figure.set_xlabel("Number of clusters")
                figure.set_ylabel("Silhouette score")
                figure.set_title("Silhouette score per number of clusters")

            experiment_results[scorer_name]["additional_plotter"] = plot_metadata

    return experiment_results


def process_for_agglo(X: npt.NDArray) -> ExperimentResult:
    params_grid = {
        "n_clusters": range(2, 30),
        "linkage": ["ward", "complete", "average", "single"],
    }

    results = defaultdict(list)
    for params in ParameterGrid(params_grid):
        model = AgglomerativeClustering(**params)
        predicted_labels = model.fit_predict(X)
        if len(set(predicted_labels)) < 2:
            continue
        for scorer_name, scorer in SCORERS.items():
            score = scorer(X, predicted_labels)
            results[scorer_name].append([score, params, predicted_labels])

    experiment_results: ExperimentResult = {}
    for scorer_name, scorer_results in results.items():
        comparator = SCORER_COMPARATORS[scorer_name]
        best_result = comparator(scorer_results, key=lambda x: x[0])
        experiment_results[scorer_name] = {
            "score": best_result[0],
            "params": best_result[1],
            "labels": best_result[2],
            "additional_plotter": blank_plotter
        }

        if scorer_name == "davies_bouldin_score":
            def plot_metadata(figure: plt.Axes) -> None:
                davies_bouldin_score_results = filter(
                    lambda result: result[1]["linkage"] == best_result[1]["linkage"],
                    results["davies_bouldin_score"]
                )
                sorted_results = sorted(davies_bouldin_score_results, key=lambda x: x[1]["n_clusters"])
                n_clusters = [result[1]["n_clusters"] for result in sorted_results]
                scores = [result[0] for result in sorted_results]
                figure.plot(n_clusters, scores, marker="o")
                figure.set_xlabel("Number of clusters")
                figure.set_ylabel("Davies-Bouldin score")
                figure.set_title("Davies-Bouldin score per number of clusters")

            experiment_results[scorer_name]["additional_plotter"] = plot_metadata

    return experiment_results


def process_for_dbscan(X: npt.NDArray, filename: str) -> ExperimentResult:
    epsilon_min, epsilon_max = EPSILON_RANGE_PER_FILE[filename]
    params_grid = {
        "eps": list(np.linspace(epsilon_min, epsilon_max, 20)),
        "min_samples": list(range(2, 21)),
    }

    results = defaultdict(list)
    for params in ParameterGrid(params_grid):
        model = DBSCAN(**params)
        predicted_labels = model.fit_predict(X)
        if len(set(predicted_labels)) < 2:
            continue
        for scorer_name, scorer in SCORERS.items():
            score = scorer(X, predicted_labels)
            results[scorer_name].append([score, params, predicted_labels])

    experiment_results: ExperimentResult = {}
    for scorer_name, scorer_results in results.items():
        comparator = SCORER_COMPARATORS[scorer_name]
        best_result = comparator(scorer_results, key=lambda x: x[0])
        experiment_results[scorer_name] = {
            "score": best_result[0],
            "params": best_result[1],
            "labels": best_result[2],
            "additional_plotter": blank_plotter
        }

        if scorer_name == "calinski_harabasz_score":
            def plot_metadata(figure: plt.Axes) -> None:
                calinski_harabasz_score_results = filter(
                    lambda result: result[1]["eps"] == best_result[1]["eps"],
                    results["calinski_harabasz_score"]
                )
                sorted_results = sorted(calinski_harabasz_score_results, key=lambda x: x[1]["min_samples"])
                min_samples = [result[1]["min_samples"] for result in sorted_results]
                scores = [result[0] for result in sorted_results]
                figure.plot(min_samples, scores, marker="o")
                figure.set_xlabel("Minimum number of samples")
                figure.set_ylabel("Calinski-Harabasz score")
                figure.set_title("Calinski-Harabasz score per minimum number of samples")

            experiment_results[scorer_name]["additional_plotter"] = plot_metadata

    return experiment_results


def process_for_hdbscan(X: npt.NDArray) -> ExperimentResult:
    params_grid = {
        "min_cluster_size": list(range(2, 21)),
        "cluster_selection_method": ["leaf", "eom"],
    }

    results = defaultdict(list)
    for params in ParameterGrid(params_grid):
        model = HDBSCAN(**params)
        predicted_labels = model.fit_predict(X)
        if len(set(predicted_labels)) < 2:
            continue
        for scorer_name, scorer in SCORERS.items():
            score = scorer(X, predicted_labels)
            results[scorer_name].append([score, params, predicted_labels])

    experiment_results: ExperimentResult = {}
    for scorer_name, scorer_results in results.items():
        comparator = SCORER_COMPARATORS[scorer_name]
        best_result = comparator(scorer_results, key=lambda x: x[0])
        experiment_results[scorer_name] = {
            "score": best_result[0],
            "params": best_result[1],
            "labels": best_result[2],
            "additional_plotter": blank_plotter
        }

        if scorer_name == "calinski_harabasz_score":
            def plot_metadata(figure: plt.Axes) -> None:
                calinski_harabasz_score_results = filter(
                    lambda result: result[1]["cluster_selection_method"] == best_result[1]["cluster_selection_method"],
                    results["calinski_harabasz_score"]
                )
                sorted_results = sorted(calinski_harabasz_score_results, key=lambda x: x[1]["min_cluster_size"])
                min_cluster_sizes = [result[1]["min_cluster_size"] for result in sorted_results]
                scores = [result[0] for result in sorted_results]
                figure.plot(min_cluster_sizes, scores, marker="o")
                figure.set_xlabel("Minimum cluster size")
                figure.set_ylabel("Calinski-Harabasz score")
                figure.set_title("Calinski-Harabasz score per minimum cluster size")

            experiment_results[scorer_name]["additional_plotter"] = plot_metadata

    return experiment_results


def run_experiment(method_type: MethodType, filename: str, X: npt.NDArray) -> ExperimentResult:
    match method_type:
        case "k-means":
            return process_for_kmeans(X)
        case "agglo":
            return process_for_agglo(X)
        case "dbscan":
            return process_for_dbscan(X, filename)
        case "hdbscan":
            return process_for_hdbscan(X)
        case _:
            raise ValueError(f"Unknown method type: {method_type}")


def process_file(method_type: MethodType, filename: str) -> None:
    X, y, X_columns, y_column = parse_file(filename + ".arff")

    experiment_result = run_experiment(method_type, filename, X)

    PLOTS_DIR.mkdir(exist_ok=True)

    directory = PLOTS_DIR / filename
    directory.mkdir(exist_ok=True)

    for scorer_name, prediction in experiment_result.items():
        figure = plot_result(method_type, X_columns, X, y_column, y, prediction, scorer_name)
        figure.tight_layout()
        scorer_directory = directory / scorer_name
        scorer_directory.mkdir(exist_ok=True)
        figure.savefig(scorer_directory / f"{method_type}.png")
        plt.close(figure)


if __name__ == "__main__":
    test_cases: List[Tuple[MethodType, List[str]]] = [
        ("k-means", ["2d-10c", "diamond9", "banana", "twenty"]),
        ("agglo", ["2d-10c", "diamond9", "banana", "twenty"]),
        ("dbscan", ["jain", "spiral", "diamond9", "atom", "twenty"]),
        ("hdbscan", ["dpb", "cluto-t4-8k", "jain", "diamond9", "banana", "ds2c2sc13"]),
    ]

    for method_type_, arff_files in test_cases:
        print(f"Processing {method_type_}...")
        for arff_file in arff_files:
            print(f"--> {arff_file}")
            process_file(method_type_, arff_file)
