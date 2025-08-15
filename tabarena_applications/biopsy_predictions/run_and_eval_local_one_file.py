"""Example code to run and evaluate several SOTA ML models on a custom dataset using TabArena.

This is an example of how to run all experiments in sequence on a custom dataset.
If one wants to run many models and splits, this can take a long time.
Thus, we recommend to parallelize the runs.

The code below runs only the default configurations and the first fold of the dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabarena_applications.biopsy_predictions.get_local_task import (
    get_tasks_for_biopsie,
)
from tabrepo import EvaluationRepository
from tabrepo.benchmark.experiment import run_experiments_new
from tabrepo.models.utils import get_configs_generator_from_name
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.generate_repo import generate_repo
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena

REPO_DIR = str(Path(__file__).parent / "tabarena_out" / "repos")
"""Cache location for the aggregated results."""
TABARENA_DIR = str(Path(__file__).parent / "tabarena_out" / "custom_dataset")
"""Output directory for saving the results and result artifacts from TabArena."""
EVAL_DIR = str(Path(__file__).parent / "tabarena_out" / "evals")
"""Output for artefacts from the evaluation results of the custom model."""

def run_tabarena_with_custom_dataset() -> None:
    """Run TabArena on a custom dataset."""
    tasks = [get_tasks_for_biopsie()]

    # Number of random configurations to generate for each model.
    # set to larger than 0 to get tuning and ensembling results.
    num_random_configs = 0
    model_names = [
        "RealMLP",
        "TabM",
        "ModernNCA",
        "TabDPT",
        "TabICL",
        "TabPFNv2",
        "Mitra",
        "CatBoost",
        "EBM",
        "ExtraTrees",
        "KNN",
        "LightGBM",
        "Linear",
        "RandomForest",
        "XGBoost",
    ]

    model_experiments = []
    for model_name in model_names:
        config_generator = get_configs_generator_from_name(model_name)
        model_experiments.extend(
            config_generator.generate_all_bag_experiments(
                num_random_configs=num_random_configs
            )
        )

    run_experiments_new(
        output_dir=TABARENA_DIR,
        model_experiments=model_experiments,
        tasks=tasks,
        repetitions_mode="matrix",
        # run 1 fold, increase this to run more folds -> (3, 10)
        repetitions_mode_args=(1, 1),
    )


def run_example_for_evaluate_results_on_custom_dataset() -> None:
    """Example for evaluating the cached results with similar plots to the TabArena paper."""
    clf_task = get_tasks_for_biopsie()

    task_metadata = load_task_metadata(paper=True)
    task_metadata = pd.DataFrame(columns=task_metadata.columns)
    task_metadata["tid"] = [clf_task.task_id]
    task_metadata["name"] = [clf_task.tabarena_task_name]
    task_metadata["task_type"] = ["Supervised Classification"]
    task_metadata["dataset"] = [
        clf_task.tabarena_task_name,
    ]
    task_metadata["NumberOfInstances"] = [
        len(clf_task._dataset),
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
    list(df_results["config_type"].unique())
    df_results = PaperRunTabArena.compute_normalized_error_dynamic(
        df_results=df_results
    )
    df_results.to_csv(Path(EVAL_DIR) / "results.csv")


def run_simple_plot():
    df = pd.read_csv(Path(EVAL_DIR) / "results.csv", index_col=0)

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    plt.figure(figsize=(10, 6))
    df = df[~df["method_subtype"].isin(["tuned", "tuned_ensemble"])]
    df["metric_error"] = 1 - df["metric_error"]
    df["metric_error_val"] = 1 - df["metric_error_val"]
    df = df.sort_values(by="metric_error", ascending=False)
    sns.barplot(
        data=df,
        x="metric_error",
        y="framework",
    )
    plt.xlabel("ROC AUC")
    plt.ylabel("Framework")
    plt.xlim(0.5)
    plt.tight_layout()
    plt.savefig(Path(EVAL_DIR) / "./results.pdf")
    plt.show()

    sns.barplot(
        data=df,
        x="metric_error",
        y="framework",
    )
    plt.xlabel("Validation ROC AUC")
    plt.ylabel("Framework")
    plt.xlim(0.5)
    plt.tight_layout()
    plt.savefig(Path(EVAL_DIR) / "results_val.pdf")
    plt.show()


if __name__ == "__main__":
    run_tabarena_with_custom_dataset()
    run_example_for_evaluate_results_on_custom_dataset()
    run_simple_plot()
