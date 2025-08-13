"""Code to use pre-defined benchmark enviroment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# # --- Mitra Run (12/08/2025).
# # Several jobs were stopped because of too conservative memory estimation, re-ran with 64GB RAM.
# BenchmarkSetup(
#     benchmark_name="mitra_12082025",
#     models=[
#         ("Mitra", 0, "fold-config-wise"),
#     ],
#     # memory_limit=64,
#     num_gpus=1,
#     methods_per_job=5,
#     sequential_local_fold_fitting=True,
#     setup_ray_for_slurm_shared_resources_environment=False,
# ).setup_jobs()

# # --- Test which setup to use for future TabArena runs.
# BenchmarkSetup(
#     benchmark_name="realmlp_0108_seed_fcw",
#     tabarena_lite=True,
#     models=[
#         ("RealMLP", "all", "fold-config-wise"),
#     ],
# ).setup_jobs()
# BenchmarkSetup(
#     benchmark_name="realmlp_0108_seed_fw",
#     tabarena_lite=True,
#     models=[
#         ("RealMLP", "all", "fold-wise"),
#     ],
# ).setup_jobs()
# BenchmarkSetup(
#     benchmark_name="realmlp_0108_seed_static",
#     tabarena_lite=True,
#     models=[
#         ("RealMLP", "all", "static"),
#     ],
# ).setup_jobs()

# --- Run of EBM with new search space and memory estimation (03/08/2025).
# Notes: for memory out cases, we re-ran and increased the memory limit to 96GB.
# BenchmarkSetup(
#     benchmark_name="ebm_03082025",
#     # memory_limit=96,  # in GB
#     models=[
#         ("EBM", "all", "fold-config-wise"),
#     ],
#     methods_per_job=1,
# ).setup_jobs()