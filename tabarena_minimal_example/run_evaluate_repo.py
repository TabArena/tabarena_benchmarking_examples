"""Example code to run TabArena(-Lite) experiments with a custom model."""

from __future__ import annotations

import pandas as pd

from pathlib import Path

from tabrepo import EvaluationRepository
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena
from tabrepo.nips2025_utils.load_final_paper_results import load_paper_results


def rename_default(config_type: str) -> str:
    return f"{config_type} (default)"


def evaluate_repo(repo_dir: str):
    """

    Requires first running `run_generate_repo.py`

    Parameters
    ----------
    repo_dir: str

    Returns
    -------

    """
    repo: EvaluationRepository = EvaluationRepository.from_dir(repo_dir)
    repo.set_config_fallback(repo.configs()[0])  # tmp: need to set a fallback config in case missing results are present

    plotter = PaperRunTabArena(repo=repo, output_dir="example_artifacts", backend="native")
    df_results = plotter.run_no_sim()

    is_default = df_results["framework"].str.contains("_c1_") & (df_results["method_type"] == "config")
    df_results.loc[is_default, "framework"] = df_results.loc[is_default]["config_type"].apply(rename_default)

    # todo: tasks instead of dataset, folds
    datasets = list(df_results["dataset"].unique())
    folds = list(df_results["fold"].unique())

    config_types = list(df_results["config_type"].unique())

    df_results_w_norm_err, df_results_holdout_w_norm_err, datasets_tabpfn, datasets_tabicl = load_paper_results(
        load_from_s3=True,  # Set to false in future runs for faster runtime
    )
    df_results_w_norm_err = df_results_w_norm_err[df_results_w_norm_err["fold"].isin(folds)]
    df_results_w_norm_err = df_results_w_norm_err[df_results_w_norm_err["dataset"].isin(datasets)]

    df_results = pd.concat([df_results, df_results_w_norm_err], ignore_index=True)
    df_results = PaperRunTabArena.compute_normalized_error_dynamic(df_results=df_results)

    plotter.eval(
        df_results=df_results,
        framework_types_extra=config_types,
    )


if __name__ == "__main__":
    evaluate_repo(
        repo_dir=str(Path(__file__).parent / "repos" / "ExampleRepo"),
    )
