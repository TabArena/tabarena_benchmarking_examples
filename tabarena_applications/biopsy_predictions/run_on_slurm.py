"""Setup slurm jobs for a local task for the collaborations."""

from __future__ import annotations

import pandas as pd
from tabarena_applications.biopsy_predictions.get_local_task import (
    get_tasks_for_biopsie,
)
from tabflow_slurm.setup_slurm_base import BenchmarkSetup


def _get_metadata(task_id_str) -> pd.DataFrame:
    return pd.DataFrame(
        [[10, 3, task_id_str, True, True, "binary"]],
        columns=[
            "tabarena_num_repeats",
            "num_folds",
            "task_id",
            "can_run_tabicl",
            "can_run_tabpfnv2",
            "problem_type",
        ],
    )


# --- Run Local Dataset Benchmark for Biopsy collaboration
user_task = get_tasks_for_biopsie(dataset_file="biopsie_preprocessed_full_cohort.csv")
BenchmarkSetup(
    custom_metadata=_get_metadata(task_id_str=user_task.task_id_str),
    benchmark_name="biopsie_preprocessed_full_cohort",
    n_random_configs=25,
    models=[
        # -- TFMs
        ("TabPFNv2", "all", "fold-config-wise"),
        # -- Neural networks
        ("RealMLP", "all", "fold-config-wise"),
        ("TabM", "all", "fold-config-wise"),
        # -- Tree-based models
        ("CatBoost", "all", "fold-config-wise"),
        ("EBM", "all", "fold-config-wise"),
        ("LightGBM", "all", "fold-config-wise"),
        ("RandomForest", "all", "fold-config-wise"),
        # -- Baselines
        ("KNN", "all", "fold-config-wise"),
        ("Linear", "all", "fold-config-wise"),
    ],
    num_gpus=1,
    methods_per_job=5,
).setup_jobs()


# local cohort
user_task = get_tasks_for_biopsie(dataset_file="biopsie_preprocessed_local_cohort.csv")
BenchmarkSetup(
    custom_metadata=_get_metadata(task_id_str=user_task.task_id_str),
    benchmark_name="biopsie_preprocessed_local_cohort",
    n_random_configs=25,
    models=[
        ("TabPFNv2", "all", "fold-config-wise"),
        ("RandomForest", "all", "fold-config-wise"),
        ("Linear", "all", "fold-config-wise"),
    ],
    num_gpus=1,
    methods_per_job=5,
).setup_jobs()

# no prior surveillance cohort
user_task = get_tasks_for_biopsie(
    dataset_file="biopsie_preprocessed_no_prior_surveillance_cohort.csv"
)
BenchmarkSetup(
    custom_metadata=_get_metadata(task_id_str=user_task.task_id_str),
    benchmark_name="biopsie_preprocessed_no_prior_surveillance_cohort",
    n_random_configs=25,
    models=[
        ("TabPFNv2", "all", "fold-config-wise"),
        ("RandomForest", "all", "fold-config-wise"),
        ("Linear", "all", "fold-config-wise"),
    ],
    num_gpus=1,
    methods_per_job=5,
).setup_jobs()
