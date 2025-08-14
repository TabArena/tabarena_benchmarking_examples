"""Setup slurm jobs for a local task for the collaborations."""

from __future__ import annotations

import pandas as pd
from tabarena_applications.biopsie_predictions.get_local_task import (
    get_tasks_for_biopsie,
)
from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# --- Run Local Dataset Benchmark for Biopsy collaboration
user_task = get_tasks_for_biopsie()
custom_metadata = pd.DataFrame(
    [[10, 3, user_task.task_id_str, True, True, "binary"]],
    columns=[
        "tabarena_num_repeats",
        "num_folds",
        "task_id",
        "can_run_tabicl",
        "can_run_tabpfnv2",
        "problem_type",
    ],
)
# N-many jobs: 12*10*3*51 + 3*10*3*1
BenchmarkSetup(
    custom_metadata=custom_metadata,
    benchmark_name="biopsy_13082025",
    n_random_configs=50,
    models=[
        # -- TFMs
        ("TabDPT", 0, "fold-config-wise"),
        ("TabICL", 0, "fold-config-wise"),
        ("Mitra", 0, "fold-config-wise"),
        ("TabPFNv2", "all", "fold-config-wise"),
        # -- Neural networks
        ("RealMLP", "all", "fold-config-wise"),
        ("ModernNCA", "all", "fold-config-wise"),
        ("TabM", "all", "fold-config-wise"),
        # -- Tree-based models
        ("CatBoost", "all", "fold-config-wise"),
        ("EBM", "all", "fold-config-wise"),
        ("ExtraTrees", "all", "fold-config-wise"),
        ("LightGBM", "all", "fold-config-wise"),
        ("RandomForest", "all", "fold-config-wise"),
        ("XGBoost", "all", "fold-config-wise"),
        # -- Baselines
        ("KNN", "all", "fold-config-wise"),
        ("Linear", "all", "fold-config-wise"),
    ],
    num_gpus=1,
    methods_per_job=5,
).setup_jobs()
