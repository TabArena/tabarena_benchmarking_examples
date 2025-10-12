"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup


# # --- Checking classification bug with ModernNCA (02/10/2025).
# BenchmarkSetup(
#     benchmark_name="mnca_02102025",
#     models=[
#         ("ModernNCA", "all", "static"),
#     ],
#     num_gpus=1,
#     tabarena_lite=True,
#     problem_types_to_run=[
#         "binary",
#         "multiclass",
#     ],
#     methods_per_job=10,
# ).setup_jobs()

# # --- xRFM For TabArena-Full
# BenchmarkSetup(
#     benchmark_name="xrfm_08092025",
#     models=[
#         ("xRFM", "all", "fold-config-wise"),
#     ],
#     num_gpus=1,
#     methods_per_job=10,
#     memory_limit=40, # for VRAM estimate
# ).setup_jobs()

# # --- LimiX (ran only for TabPFN-subset)
# BenchmarkSetup(
#     benchmark_name="limix_04092025_no_retrieval",
#     models=[
#         ("LimiX", 0, "fold-config-wise"),
#     ],
#     num_gpus=1,
# ).setup_jobs()

# # --- RealMLP on GPU and with new search space (15/08/2025).
# BenchmarkSetup(
#     benchmark_name="realmlp_15082025",
#     models=[
#         ("RealMLP", "all", "fold-config-wise"),
#     ],
#     num_gpus=1,
#     methods_per_job=5,
# ).setup_jobs()

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
