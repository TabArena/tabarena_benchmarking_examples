"""Code to use pre-defined benchmark enviroment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

BenchmarkSetup(
    benchmark_name="realmlp_0108_seed_fcw",
    tabarena_lite=True,
    models=[
        ("RealMLP", "all", "fold-config-wise"),
    ],
).setup_jobs()
