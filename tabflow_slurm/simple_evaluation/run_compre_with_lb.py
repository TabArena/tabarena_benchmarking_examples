"""Example code to evaluate a model by comparing it to the leaderboard for TabArena(-Lite).

Before using this code, you must first run `run_tabarena_lite.py` to generate the input files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tabflow_slurm.run_setup_slurm_jobs import BenchmarkSetupGPUModels
from tabrepo import EvaluationRepository
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.generate_repo import generate_repo
from tabrepo.nips2025_utils.load_final_paper_results import load_paper_results
from tabrepo.paper.paper_runner_tabarena import PaperRunTabArena

BENCHMARK_DATA = BenchmarkSetupGPUModels()
REPO_DIR = str(Path(__file__).parent / "repos" / "lb_eval")
EVAL_DIR = str(Path(__file__).parent / "evals" / "lb_eval")


def rename_default(config_type: str) -> str:
    return f"{config_type} (default)"


# def rename_new_framework():
#     # Rename
#     df_results["config_type"] = df_results["config_type"].str.replace(
#         "REALMLP", "REALMLP-GPU-RS"
#     )
#     df_results["framework"] = (
#         df_results["framework"]
#         .str.replace("RealMLP", "RealMLP-GPU-RS")
#         .str.replace("REALMLP", "REALMLP-GPU-RS")
#     )


def compare_to_lb():
    task_metadata = load_task_metadata(paper=True)
    repo: EvaluationRepository = generate_repo(
        experiment_path=BENCHMARK_DATA.output_dir, task_metadata=task_metadata
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
    ].apply(rename_default)
    datasets = list(df_results["dataset"].unique())
    folds = list(df_results["fold"].unique())
    # rename_new_framework()
    config_types = list(df_results["config_type"].unique())
    df_results_gpu_rs = df_results.copy()

    # Results paper
    df_results, _, _, _ = load_paper_results(
        load_from_s3=False,  # Set to false in future runs for faster runtime
    )
    df_results = df_results[
        df_results["fold"].isin(folds) & df_results["dataset"].isin(datasets)
    ]
    df_results_paper = df_results.copy()

    # Merge results
    df_results = PaperRunTabArena.compute_normalized_error_dynamic(
        df_results=pd.concat([df_results_gpu_rs, df_results_paper], ignore_index=True)
    )

    # Eval
    plotter.eval(
        df_results=df_results,
        framework_types_extra=config_types,
        plot_extra_barplots=True,
        show=True,
        imputed_names=["TabPFNv2"],
    )


if __name__ == "__main__":
    compare_to_lb()
