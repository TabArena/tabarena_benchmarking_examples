"""Code with hardcoded benchmark setup for generating input data to our SLURM job submission script.

Running this code will generate `slurm_run_data.json` with all the data required to run the array jobs
via `submit_template_gpu.sh`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from tabrepo.utils.cache import CacheFunctionPickle
from tqdm import tqdm

BASE_PATH = "/work/dlclarge2/purucker-tabarena/"
"""Base path for the project, code, and results. Within this directory, all results, code, and logs for TabArena will
be saved. Adjust below as needed if more than one base path is desired. On a typical SLURM system, this base path
should point to a persistent workspace that all your jobs can access.

For our system, we used a structure as follows:
    - BASE_PATH
        - code              -- contains all code for the project
        - venvs             -- contains all virtual environments
        - input_data        -- contains all input data (e.g. TabRepo artifacts), this is not used so far
        - output            -- contains all output data from running benchmark
        - slurm_out         -- contains all SLURM output logs
        - .openml-cache     -- contains the OpenML cache
"""


@dataclass
class BenchmarkSetupGPUModels:
    """Manually set the parameters for the benchmark run."""

    metadata: str = (
        BASE_PATH + "code/tabarena/tabflow_slurm/tabarena_dataset_metadata.csv"
    )
    """Dataset/task Metadata for TabArena, download this csv from: https://github.com/TabArena/dataset_curation/blob/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv
    Adjust as needed to run less datasets/tasks or create a new constraint used for filtering."""

    configs: str = BASE_PATH + "code/tabarena/tabflow_slurm/models/configs_all_gpu.yaml"
    """YAML file with the configs to run. See ./models/run_generate_tabarena_gpu_configs.py for
    how to generate this file."""

    python: str = BASE_PATH + "venvs/tabarena_gpu/bin/python"
    """Python executable and environment to use for the SLURM jobs. This should point to a Python
    executable within a (virtual) environment."""

    run_script: str = (
        BASE_PATH + "code/tabarena/tabflow_slurm/run_tabarena_experiment.py"
    )
    """Python script to run the benchmark. This should point to the script that runs the benchmark
    for TabArena."""

    openml_cache: str = BASE_PATH + ".openml-cache"
    """OpenML cache directory. This is used to store dataset and tasks data from OpenML."""
    tabrepo_cache_dir: str = BASE_PATH + "input_data/tabrepo"
    """TabRepo cache directory."""
    output_dir: str = BASE_PATH + "output/gpu_runs"
    """Output directory for the benchmark."""

    slurm_script: str = "submit_template_gpu.sh"
    """Name of the SLURM (array) script that to run on the cluster (only used to print the command
     to run)."""
    slurm_log_output: str = BASE_PATH + "slurm_out/gpu_runs"
    """Directory for the SLURM output logs. This is used to store the output logs from the
    SLURM jobs."""

    num_cpus: int = 8
    """Number of CPUs to use for the SLURM jobs. The number of CPUs available on the node and in
    sync with the slurm_script."""
    num_gpus: int = 1
    """Number of GPUs to use for the SLURM jobs. The number of GPUs available on the node and in
    sync with the slurm_script."""
    memory_limit: int = 32
    """Memory limit for the SLURM jobs. The memory limit available on the node and in sync with
    the slurm_script."""

    methods_per_job: int = 5
    """Batching of several experiments per job. This is used to reduce the number of SLURM jobs.
    Adjust the time limit in the slurm_script accordingly."""
    sequential_local_fold_fitting: bool = (
        True  # Do not use Ray for GPU-fitting for now.
    )
    """Use Ray for local fold fitting. This is used to speed up the local fold fitting. For CPU
    runs, or if multiple GPUs are available, this should be set to False"""
    setup_ray_for_slurm_shared_resources_environment: bool = False
    """Prepare Ray for a SLURM shared resource environment. This is used to setup Ray for SLURM
    shared resources. Recommended to set to True if sequential_local_fold_fitting is False."""

    tabarena_lite: bool = False
    """Run only TabArena-Lite, that is: only the first split of each dataset, and the default
    configuration and up to `tabarena_lite_n_configs` random configs."""
    tabarena_lite_n_configs: int = 25
    """Limit the number of random configs to run per model class in TabArena-Lite."""

    ignore_cache: bool = False
    """If True, will overwrite the cache and run all jobs again."""

    cache_cls: CacheFunctionPickle = CacheFunctionPickle
    """How to save the cache. Pickle is the current recommended default. This option and the two
    below must be in sync with the cache method in run_script."""
    cache_cls_kwargs: dict = field(
        default_factory=lambda: {"include_self_in_call": True}
    )
    """Arguments for the cache class. This is used to setup the cache class for the benchmark."""
    cache_path_format: str = "name_first"
    """Path format for the cache. This is used to setup the cache path format for the benchmark."""

    def get_jobs_to_run(self):
        """Determine all jobs to run by checking the cache and filtering invalid jobs."""
        from tabrepo.benchmark.experiment.experiment_utils import check_cache_hit

        Path(self.openml_cache).mkdir(parents=True, exist_ok=True)
        Path(self.tabrepo_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.slurm_log_output).mkdir(parents=True, exist_ok=True)

        metadata = pd.read_csv(self.metadata)
        # Read YAML file and get the number of configs
        with open(self.configs) as file:
            configs = yaml.safe_load(file)["methods"]

        if self.tabarena_lite:
            # Keep first default and the 25 first random configs per model class
            # -> Assumes name suffixes have been set correctly for the configs!
            assert all(
                "name_suffix" in e.get("model_hyperparameters", {}).get("ag_args", {})
                for e in configs
            ), (
                f"All configs should have a name_suffix in the model_hyperparameters.ag_args in the YAML files, please update: {self.configs}."
            )
            configs = [
                config
                for config in configs
                if (
                    (config["model_hyperparameters"]["ag_args"]["name_suffix"] == "_c1")
                    or (
                        (
                            config["model_hyperparameters"]["ag_args"][
                                "name_suffix"
                            ].startswith("_r")
                        )
                        and (
                            int(
                                config["model_hyperparameters"]["ag_args"][
                                    "name_suffix"
                                ][2:]
                            )
                            <= self.tabarena_lite_n_configs
                        )
                    )
                )
            ]

        jobs = []
        total_jobs = (
            metadata["num_folds"] * metadata["tabarena_num_repeats"]
        ).sum() * len(configs)
        to_run_jobs = 0
        pbar = tqdm(desc="Checking Cache and Filter Invalid Jobs", total=total_jobs)
        for row in metadata.itertuples():
            repeats_folds = product(
                range(int(row.tabarena_num_repeats)), range(int(row.num_folds))
            )
            if self.tabarena_lite:  # Filter to only first split.
                repeats_folds = list(repeats_folds)[:1]

            for repeat_i, fold_i in repeats_folds:
                all_configs = list(enumerate(configs))
                while len(all_configs) > 0:
                    config_batch = []
                    while (len(config_batch) < self.methods_per_job) and (
                        len(all_configs) > 0
                    ):
                        config_index, config = all_configs.pop(0)

                        # Filter out-of-constraints datasets
                        if (
                            # Skip TabICL if the dataset cannot run it
                            (
                                (config["model_cls"] == "TabICLModel")
                                and (not row.can_run_tabicl)
                            )
                            # Skip TabPFN if the dataset cannot run it
                            or (
                                (config["model_cls"] == "TabPFNV2Model")
                                and (not row.can_run_tabpfnv2)
                            )
                        ):
                            # Reset total_jobs and update progress bar
                            total_jobs -= 1
                            pbar.total = total_jobs
                            pbar.refresh()
                            continue

                        if not check_cache_hit(
                            result_dir=self.output_dir,
                            method_name=config["name"],
                            task_id=row.task_id,
                            fold=fold_i,
                            repeat=repeat_i,
                            cache_path_format=self.cache_path_format,
                            cache_cls=self.cache_cls,
                            cache_cls_kwargs=self.cache_cls_kwargs,
                            mode="local",
                        ):
                            to_run_jobs += 1
                            config_batch.append(config_index)

                        pbar.update(1)

                    if config_batch:
                        jobs.append(
                            {
                                "dataset_name": row.dataset_name,
                                "task_id": row.task_id,
                                "fold": fold_i,
                                "repeat": repeat_i,
                                "config_index": config_batch,
                            },
                        )
        print(f"Generated {to_run_jobs}/{total_jobs} jobs to run without batching.")
        print(f"Jobs with batching: {len(jobs)}")
        return jobs

    def get_jobs_dict(self):
        jobs = list(self.get_jobs_to_run())
        default_args = {
            "python": self.python,
            "run_script": self.run_script,
            "openml_cache_dir": self.openml_cache,
            "configs_yaml_file": self.configs,
            "tabrepo_cache_dir": self.tabrepo_cache_dir,
            "output_dir": self.output_dir,
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory_limit": self.memory_limit,
            "setup_ray_for_slurm_shared_resources_environment": self.setup_ray_for_slurm_shared_resources_environment,
            "ignore_cache": self.ignore_cache,
            "sequential_local_fold_fitting": self.sequential_local_fold_fitting,
        }
        return {"defaults": default_args, "jobs": jobs}


def setup_jobs():
    bench = BenchmarkSetupGPUModels()
    jobs_dict = bench.get_jobs_dict()
    n_jobs = len(jobs_dict["jobs"])
    if n_jobs == 0:
        print("No jobs to run.")
        return

    with open("slurm_run_data.json", "w") as f:
        json.dump(jobs_dict, f)

    print("Run the following command to start the jobs:")
    print(f"sbatch --array=0-{n_jobs - 1}%100 {bench.slurm_script}")


if __name__ == "__main__":
    setup_jobs()
