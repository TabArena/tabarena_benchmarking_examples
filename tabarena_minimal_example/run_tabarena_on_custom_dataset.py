"""Example code to benchmark models on a new custom (non-OpenML) dataset with TabArena."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tabrepo.benchmark.experiment import run_experiments_new
from tabrepo.benchmark.task import UserTask

REPO_DIR = str(Path(__file__).parent / "repos" / "custom_dataset")
"""Cache location for the aggregated results."""

TABARENA_DIR = str(Path(__file__).parent / "tabarena_out" / "custom_dataset")
"""Output directory for saving the results and result artifacts from TabArena."""

EVAL_DIR = str(Path(__file__).parent / "evals" / "custom_dataset")
"""Output for artefacts from the evaluation results of the custom model."""


def get_custom_classification_task() -> UserTask:
    """Example for defining a custom classification task/dataset to run with for TabArena."""
    # Create toy classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    # Add cat features
    cats_1 = ["a"] * 25 * 5 + ["b"] * 25 * 5 + ["c"] * 25 * 5 + ["d"] * 25 * 5
    cats_2 = ["x"] * 34 * 5 + ["y"] * 33 * 5 + ["z"] * 33 * 5
    # Add nan values
    cats_1[0] = np.nan
    cats_1[49] = np.nan
    X.iloc[0, 2] = np.nan
    X.iloc[0, 3] = np.nan
    X = X.assign(cat_1=pd.Categorical(cats_1), cat_2=pd.Categorical(cats_2))
    dataset = pd.concat([X, y.rename("target")], axis=1)

    # Create a stratified 10-repeated 3-fold split (any other split can be used as well)
    n_repeats, n_splits = 10, 3
    sklearn_splits = RepeatedStratifiedKFold(
        n_repeats=n_repeats, n_splits=n_splits, random_state=42
    ).split(X=dataset.drop(columns=["target"]), y=dataset["target"])
    # Transform the splits into a standard dictionary format expected by TabArena
    splits = {
        split_i // n_repeats: {
            split_i % n_splits: (train_idx.tolist(), test_idx.tolist())
        }
        for split_i, (train_idx, test_idx) in enumerate(sklearn_splits)
    }

    return UserTask(
        task_name="ToyClf",
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=splits,
    )


def get_custom_regression_task() -> UserTask:
    """Example for defining a custom regression task/dataset to run with for TabArena."""
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=10,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    # Add cat features
    cats_1 = ["a"] * 25 * 5 + ["b"] * 25 * 5 + ["c"] * 25 * 5 + ["d"] * 25 * 5
    cats_2 = ["x"] * 34 * 5 + ["y"] * 33 * 5 + ["z"] * 33 * 5
    # Add nan values
    cats_1[0] = np.nan
    cats_1[49] = np.nan
    X.iloc[0, 2] = np.nan
    X.iloc[0, 3] = np.nan
    X = X.assign(cat_1=pd.Categorical(cats_1), cat_2=pd.Categorical(cats_2))
    dataset = pd.concat([X, y.rename("target")], axis=1)

    # Create a holdout split without repeats
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))), test_size=0.33, random_state=42, shuffle=True
    )
    # Transform the splits into a standard dictionary format expected by TabArena
    splits = {0: {0: (train_idx, test_idx)}}

    return UserTask(
        task_name="ToyReg",
        dataset=dataset,
        target_feature="target",
        problem_type="regression",
        splits=splits,
    )


def get_model_configs_to_benchmark(
    model_names: list[str] | None,
    *,
    num_random_configs: int = 1,
    custom_rf: bool = True,
    sequential_fold_fitting: bool = True,
) -> list:
    """Get the configurations for all models you want to benchmark.


    Parameters
    ----------
    model_names : list of str or None
        A list of model names to benchmark. If None, all models from TabArena-v0.1 are used.
    num_random_configs : int
        The number of random configurations to generate for each model.
    custom_rf : bool
        If True, includes the custom random forest model in the list of configurations.
        This model is defined in `custom_tabarena_model.py`.

    Returns:
    -------
    experiments_lst : list
        A list of configurations for TabArena.
    """
    from tabrepo.models.utils import get_configs_generator_from_name

    if model_names is None:
        model_names = [
            "RealMLP",
            "TabM",
            "ModernNCA",
            "TabDPT",
            "TabICL",
            "TabPFNv2",
            "CatBoost",
            "EBM",
            "ExtraTrees",
            "FastaiMLP",
            "KNN",
            "LightGBM",
            "Linear",
            "TorchMLP",
            "RandomForest",
            "XGBoost",
        ]

    # Get all the models from TabArena-v0.1
    experiments_lst = []
    for model_name in model_names:
        config_generator = get_configs_generator_from_name(model_name)
        experiments_lst.extend(
            config_generator.generate_all_bag_experiments(
                num_random_configs=num_random_configs
            )
        )

    # Get our custom random forest model
    if custom_rf:
        from custom_tabarena_model import get_configs_for_custom_rf

        experiments_lst.extend(
            get_configs_for_custom_rf(
                default_config=True, num_random_configs=num_random_configs
            )
        )

    if sequential_fold_fitting:
        for m_i in range(len(experiments_lst)):
            if (
                "ag_args_ensemble"
                not in experiments_lst[m_i].method_kwargs["model_hyperparameters"]
            ):
                experiments_lst[m_i].method_kwargs["model_hyperparameters"][
                    "ag_args_ensemble"
                ] = {}
            experiments_lst[m_i].method_kwargs["model_hyperparameters"][
                "ag_args_ensemble"
            ]["fold_fitting_strategy"] = "sequential_local"

    return experiments_lst


def run_tabarena_with_custom_dataset() -> None:
    """Run TabArena on a custom dataset."""
    # Get all tasks from TabArena-v0.1
    tasks = [get_custom_classification_task(), get_custom_regression_task()]

    # Gets 1 default and 1 random config
    model_experiments = get_model_configs_to_benchmark(
        model_names=["Linear", "KNN"],
        num_random_configs=1,
        custom_rf=True,  # Example of including a custom model in your benchmark.
    )

    run_experiments_new(
        output_dir=TABARENA_DIR,
        model_experiments=model_experiments,
        tasks=tasks,
        repetitions_mode="TabArena-Lite",
    )


def run_example_for_evaluate_results_on_custom_dataset() -> None:
    """Example for evaluating the cached results with similar plots to the TabArena paper."""
    from tabrepo import EvaluationRepository
    from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
    from tabrepo.nips2025_utils.generate_repo import generate_repo
    from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena

    clf_task, reg_task = get_custom_classification_task(), get_custom_regression_task()

    # TODO: improve how users can pass only the required metadata to the eval code.
    task_metadata = load_task_metadata(paper=True)
    task_metadata = pd.DataFrame(columns=task_metadata.columns)
    task_metadata["tid"] = [clf_task.task_id, reg_task.task_id]
    task_metadata["name"] = [clf_task.tabarena_task_name, reg_task.tabarena_task_name]
    task_metadata["task_type"] = ["Supervised Classification", "Supervised Regression"]
    task_metadata["dataset"] = [
        clf_task.tabarena_task_name,
        reg_task.tabarena_task_name,
    ]
    task_metadata["NumberOfInstances"] = [
        len(clf_task._dataset),
        len(reg_task._dataset),
    ]

    repo: EvaluationRepository = generate_repo(
        experiment_path=TABARENA_DIR, task_metadata=task_metadata
    )
    repo.to_dir(REPO_DIR)
    repo: EvaluationRepository = EvaluationRepository.from_dir(REPO_DIR)
    repo.set_config_fallback(repo.configs()[0])

    plotter = PaperRunTabArena(repo=repo, output_dir=EVAL_DIR, backend="native")
    df_results = plotter.run_no_sim()

    is_default = df_results["framework"].str.contains("_c1_") & (
        df_results["method_type"] == "config"
    )
    df_results.loc[is_default, "framework"] = df_results.loc[is_default][
        "config_type"
    ].apply(lambda c: f"{c} (default)")

    config_types = list(df_results["config_type"].unique())

    # df_results now has all the results one could use for plotting.
    df_results = PaperRunTabArena.compute_normalized_error_dynamic(
        df_results=df_results
    )

    # Create plots with the eval code from the paper.
    # Saves results to the ./custom_dataset_eval/ directory.
    plotter.eval(
        df_results=df_results,
        framework_types_extra=config_types,
        baselines=None,
        task_metadata=task_metadata,
        calibration_framework="LR (default)",
        plot_cdd=False,
    )


if __name__ == "__main__":
    run_tabarena_with_custom_dataset()
    run_example_for_evaluate_results_on_custom_dataset()
