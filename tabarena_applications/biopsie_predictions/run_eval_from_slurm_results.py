from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabrepo import EvaluationRepository
from tabrepo.benchmark.task import UserTask
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.generate_repo import generate_repo
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena

EVAL_DIR = str(Path(__file__).parent / "tabarena_out" / "slurm_results")
REPO_DIR = str(Path(__file__).parent / "tabarena_out" / "repos")
TABARENA_DIR = "/work/dlclarge2/purucker-tabarena/output/biopsy_13082025"
UserTaskStr = "UserTask|1494229299|BiopsieCancerPrediction|/work/dlclarge2/purucker-tabarena/code/tabarena_benchmarking_examples/tabarena_applications/biopsie_predictions/tabarena_out/local_tasks"


def run_example_for_evaluate_results_on_custom_dataset() -> None:
    """Example for evaluating the cached results with similar plots to the TabArena paper."""
    clf_task = UserTask.from_task_id_str(UserTaskStr)
    task_metadata = load_task_metadata(paper=True)
    task_metadata = pd.DataFrame(columns=task_metadata.columns)
    task_metadata["tid"] = [clf_task.task_id]
    task_metadata["name"] = [clf_task.tabarena_task_name]
    task_metadata["task_type"] = ["Supervised Classification"]
    task_metadata["dataset"] = [
        clf_task.tabarena_task_name,
    ]
    task_metadata["NumberOfInstances"] = [2466]
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
    plotter.eval(
        df_results=df_results,
        framework_types_extra=list(df_results["config_type"].unique()),
        baselines=None,
        task_metadata=task_metadata,
        calibration_framework="RF (default)",
        plot_cdd=False,
    )



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
    plt.xlim(0.6)
    plt.tight_layout()
    plt.savefig(Path(EVAL_DIR) / "results.pdf")
    plt.show()

    sns.barplot(
        data=df,
        x="metric_error",
        y="framework",
    )
    plt.xlabel("Val ROC AUC")
    plt.ylabel("Framework")
    plt.xlim(0.5)
    plt.tight_layout()
    plt.savefig(Path(EVAL_DIR) / "results_val.pdf")
    plt.show()


if __name__ == "__main__":
    # run_example_for_evaluate_results_on_custom_dataset()
    run_simple_plot()
