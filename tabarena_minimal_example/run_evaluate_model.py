"""Example code to evaluate a model by comparing it to the leaderboard for TabArena(-Lite).

Before using this code, you must first run `run_tabarena_lite.py` to generate the input files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tabrepo import EvaluationRepository
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.generate_repo import generate_repo
from tabrepo.nips2025_utils.load_final_paper_results import load_paper_results
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena

REPO_DIR = str(Path(__file__).parent / "repos" / "ExampleRepo")
"""Cache location for the result artifacts."""


def post_process_local_results(output_dir: str):
    """Post-process the local results of TabArena and generate a result repository.

    Parameters
    ----------
    output_dir : str
        The folder location of results (the `output_dir` parameter in `run_tabarena_lite.py`).
    """
    task_metadata = load_task_metadata(paper=True)
    repo: EvaluationRepository = generate_repo(
        experiment_path=output_dir, task_metadata=task_metadata
    )
    repo.to_dir(REPO_DIR)
    return repo


def rename_default(config_type: str) -> str:
    return f"{config_type} (default)"


def evaluate_custom_model(output_dir: str):
    """Evaluate the custom model by comparing it to the leaderboard for TabArena(-Lite).

    Parameters
    ----------
    output_dir : str
        The path to the output directory where the results were saved.
    """
    post_process_local_results(output_dir=output_dir)
    repo: EvaluationRepository = EvaluationRepository.from_dir(REPO_DIR)
    repo.set_config_fallback(repo.configs()[0])

    plotter = PaperRunTabArena(
        repo=repo, output_dir="model_eval", backend="native"
    )
    df_results = plotter.run_no_sim()

    is_default = df_results["framework"].str.contains("_c1_") & (
        df_results["method_type"] == "config"
    )
    df_results.loc[is_default, "framework"] = df_results.loc[is_default][
        "config_type"
    ].apply(rename_default)
    datasets = list(df_results["dataset"].unique())
    folds = list(df_results["fold"].unique())
    config_types = list(df_results["config_type"].unique())

    df_results_w_norm_err, _, _, _ = load_paper_results(
        load_from_s3=True,  # Set to false in future runs for faster runtime
    )
    df_results_w_norm_err = df_results_w_norm_err[
        df_results_w_norm_err["fold"].isin(folds)
        & df_results_w_norm_err["dataset"].isin(datasets)
    ]
    df_results = PaperRunTabArena.compute_normalized_error_dynamic(
        df_results=pd.concat([df_results, df_results_w_norm_err], ignore_index=True)
    )

    # Saves results to the ./model_eval/ directory. Our new model is called CRF.
    plotter.eval(
        df_results=df_results,
        framework_types_extra=config_types,
    )


if __name__ == "__main__":
    evaluate_custom_model(output_dir=str(Path(__file__).parent / "tabarena_out"))
