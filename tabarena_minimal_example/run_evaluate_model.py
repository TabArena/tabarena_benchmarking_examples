"""Example code to evaluate a model by comparing it to the leaderboard for TabArena(-Lite).

Before using this code, you must first run `run_tabarena_lite.py` to generate the input files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tabrepo import EvaluationRepository
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.generate_repo import generate_repo
from tabrepo.nips2025_utils.load_final_paper_results import load_results
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena

REPO_DIR = str(Path(__file__).parent / "repos" / "custom_model")
"""Cache location for the aggregated results."""

TABARENA_DIR = str(Path(__file__).parent / "tabarena_out" / "custom_model")
"""Output directory for saving the results and result artifacts from TabArena."""

EVAL_DIR = str(Path(__file__).parent / "evals" / "custom_model")
"""Output for artefacts from the evaluation results of the custom model."""


def post_process_local_results() -> EvaluationRepository:
    """Post-process the local results of TabArena and generate a result repository.

    Parameters
    ----------
    output_dir : str
        The folder location of results (the `output_dir` parameter in `run_tabarena_lite.py`).
    """
    task_metadata = load_task_metadata(paper=True)
    repo: EvaluationRepository = generate_repo(
        experiment_path=TABARENA_DIR, task_metadata=task_metadata
    )
    repo.to_dir(REPO_DIR)

    repo = EvaluationRepository.from_dir(REPO_DIR)
    repo.set_config_fallback(repo.configs()[0])

    return repo


def load_local_results(plotter: PaperRunTabArena) -> pd.DataFrame:
    df_results = plotter.run_no_sim()
    is_default = df_results["framework"].str.contains("_c1_") & (
        df_results["method_type"] == "config"
    )
    df_results.loc[is_default, "framework"] = df_results.loc[is_default][
        "config_type"
    ].apply(lambda config_type: f"{config_type} (default)")
    return df_results


def load_paper_reuslts(df_results: pd.DataFrame):
    datasets = list(df_results["dataset"].unique())
    folds = list(df_results["fold"].unique())

    df_results_paper = load_results(lite=True)
    df_results_paper["framework"] = df_results_paper["method"]
    df_results_paper = df_results_paper[
        df_results_paper["fold"].isin(folds)
        & df_results_paper["dataset"].isin(datasets)
    ]
    df_results_paper[
        list(set(df_results.columns).difference(set(df_results_paper.columns)))
    ] = None
    df_results_paper = df_results_paper[df_results.columns]
    return PaperRunTabArena.compute_normalized_error_dynamic(
        df_results=pd.concat([df_results, df_results_paper], ignore_index=True)
    )


def evaluate_custom_model():
    """Evaluate the custom model by comparing it to the leaderboard for TabArena(-Lite).

    Parameters
    ----------
    output_dir : str
        The path to the output directory where the results were saved.
    """
    repo = post_process_local_results()
    plotter = PaperRunTabArena(repo=repo, output_dir=EVAL_DIR)

    df_results = load_local_results(plotter)
    df_results = load_paper_reuslts(df_results)

    # Saves results to the EVAL_DIR directory. Our new model is called CRF.
    plotter.eval(
        df_results=df_results,
        framework_types_extra=list(df_results["config_type"].unique()),
    )


if __name__ == "__main__":
    evaluate_custom_model()
