"""Code to use pre-defined benchmark enviroment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# Test which setup to use for future TabArena runs.
BenchmarkSetup(
    benchmark_name="realmlp_0108_seed_fcw",
    tabarena_lite=True,
    models=[
        ("RealMLP", "all", "fold-config-wise"),
    ],
).setup_jobs()
BenchmarkSetup(
    benchmark_name="realmlp_0108_seed_fw",
    tabarena_lite=True,
    models=[
        ("RealMLP", "all", "fold-wise"),
    ],
).setup_jobs()
BenchmarkSetup(
    benchmark_name="realmlp_0108_seed_static",
    tabarena_lite=True,
    models=[
        ("RealMLP", "all", "static"),
    ],
).setup_jobs()
