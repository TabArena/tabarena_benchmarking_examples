"""Code with hardcoded benchmark setup for generating input data to our SLURM job submission script.

Running this code will generate `slurm_run_data.json` with all the data required to run the array jobs
via `submit_template_gpu.sh`.

See `run_setup_slurm_jobs.py` for an example of how to use this code.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import pandas as pd
import ray
import yaml
from tabflow_slurm.ray_utils import ray_map_list, to_batch_list
from tabrepo.benchmark.experiment.experiment_utils import check_cache_hit
from tabrepo.utils.cache import CacheFunctionPickle


@dataclass
class BenchmarkSetup:
    """Manually set the parameters for the benchmark run."""

    # Required user input
    benchmark_name: str

    # Cluster Settings
    # ----------------
    base_path = "/work/dlclarge2/purucker-tabarena/"
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
    python_from_base_path: str = "venvs/tabarena_1508/bin/python"
    """Python executable and environment to use for the SLURM jobs. This should point to a Python
    executable within a (virtual) environment."""
    run_script_from_base_path: str = (
        "code/tabarena_benchmarking_examples/tabflow_slurm/run_tabarena_experiment.py"
    )
    """Python script to run the benchmark. This should point to the script that runs the benchmark
    for TabArena."""
    openml_cache_from_base_path: str = ".openml-cache"
    """OpenML cache directory. This is used to store dataset and tasks data from OpenML."""
    tabrepo_cache_dir_from_base_path: str = "input_data/tabrepo"
    """TabRepo cache directory."""
    slurm_script_cpu: str = "submit_template_cpu.sh"
    """Name of the CPU SLURM (array) script that to run on the cluster (only used to print the command
     to run)."""
    slurm_script_gpu: str = "submit_template_gpu.sh"
    """Name of the GPU SLURM (array) script that to run on the cluster (only used to print the command
     to run)."""
    slurm_log_output_from_base_path: str = "slurm_out/new_models"
    """Directory for the SLURM output logs. This is used to store the output logs from the
    SLURM jobs."""
    output_dir_base_from_base_path: str = "output/"
    """Output directory for the benchmark. In this folder a `benchmark_name` folder will be created."""
    configs_path_from_base_path: str = (
        "code/tabarena_benchmarking_examples/tabflow_slurm/benchmark_configs_"
    )
    """YAML file with the configs to run. Generated from parameters above in code below.
    File path is f"{self.base_path}{self.configs_path_from_base_path}{self.benchmark_name}.yaml"
    """

    # Task/Data Settings
    metadata_from_base_path: str = "code/tabarena_benchmarking_examples/tabflow_slurm/tabarena_dataset_metadata.csv"
    """Dataset/task Metadata for TabArena, download this csv from: https://github.com/TabArena/dataset_curation/blob/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv
    Adjust as needed to run less datasets/tasks or create a new constraint used for filtering."""
    custom_metadata: pd.DataFrame | None = None
    """Custom metadata to use for defining the tasks and datasets to run.

    The metadata must have the following columns:
        "tabarena_num_repeats": int
            The number of repeats for the task.
        "num_folds": int
            The number of folds for the task.
        "task_id": str
            The task ID for the task as an int.
            If a local task, we assume this to be `UserTask.task_id_str`.
        "can_run_tabicl": bool
            If the task can run TabICL. (<=100k training samples, <=500 features)
        "can_run_tabpfnv2" : bool
            If the task can run TabPFNv2. (<=10k training samples, <=500 features, <=10 classes)
        "problem_type": str
            The problem type of the task. Options: "binary", "regression", "multiclass"
    """
    tabarena_lite: bool = False
    """Run only TabArena-Lite, that is: only the first split of each dataset, and the default
    configuration and up to `tabarena_lite_n_configs` random configs."""
    problem_types_to_run: list[str] = field(
        # Options: "binary", "regression", "multiclass"
        default_factory=lambda: [
            "binary",
            "multiclass",
            "regression",
        ]
    )
    # Benchmark Settings
    # ------------------
    """Problem types to run in the benchmark. Adjust as needed to run only specific problem types."""
    num_cpus: int = 8
    """Number of CPUs to use for the SLURM jobs. The number of CPUs available on the node and in
    sync with the slurm_script."""
    num_gpus: int = 0
    """Number of GPUs to use for the SLURM jobs. The number of GPUs available on the node and in
    sync with the slurm_script."""
    memory_limit: int = 32
    """Memory limit for the SLURM jobs. The memory limit available on the node and in sync with
    the slurm_script."""
    n_random_configs: int = 200
    """Number of random hyperparameter configurations to run for each model"""
    models: list[tuple[str, int | str, str]] = field(default_factory=list)
    """List of models to run in the benchmark with metadata.
    Metadata keys from left to right:
        - model name: str
        - number of random hyperparameter configurations to run: int or str
            Some special cases are:
                - If 0, only the default configuration is run.
                - If "all", `n_random_configs`-many configurations are run.
        - seed method for the model: str
            - "static" for static seed across all model fits
            - "fold-wise" for different seed per fold
            - "fold-config-wise" for different seeds per fold and configuration

    Remove or comment out models to that you do not want to run.
    Options from the current state of TabArena are:
    default_factory=lambda: [
            # -- TFMs (or similar)
            ("TabDPT", 0, "static"),
            ("TabICL", "all", "static"),
            ("TabPFNv2", "all", "static"),
            ("Mitra", "all", "fold-config-wise"),
            # -- Neural networks
            ("TabM", "all", "static"),
            ("RealMLP", "all", "fold-config-wise"),
            ("ModernNCA", "all", "static"),
            ("FastaiMLP", "all", "static"),
            ("TorchMLP", "all", "static"),
            # -- Tree-based models
            ("CatBoost", "all", "static"),
            ("EBM", "all", "fold-config-wise"),
            ("ExtraTrees", "all", "static"),
            ("LightGBM", "all", "static"),
            ("RandomForest", "all", "static"),
            ("XGBoost", "all", "static"),
            # -- Baselines
            ("KNN", 50, "static"),
            ("Linear", "all", "static"),
        ]
    )
    """
    methods_per_job: int = 5
    """Batching of several experiments per job. This is used to reduce the number of SLURM jobs.
    Adjust the time limit in the slurm_script accordingly."""
    setup_ray_for_slurm_shared_resources_environment: bool = True
    """Prepare Ray for a SLURM shared resource environment. This is used to setup Ray for SLURM
    shared resources. Recommended to set to True if sequential_local_fold_fitting is False."""
    preprocessing_pieplines: list[str] = field(
        default_factory=lambda: ["default"]
    )
    """Preprocessing pipelines to add to the configurations we want to run.

    Each options multiplies the number of configurations to run by the number of
    pipelines. For example, if we have 10 configurations and 2 pipelines, we will
    run 20 configurations.

    Options:
        - "default": Use the default preprocessing pipeline.
        - Any other string registered in `tabrepo.benchmark.preprocessing.preprocessing_register`.
    """

    # Misc Settings
    # -------------
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
    num_ray_cpus = 8
    """Number of CPUs to use for checking the cache and generating the jobs. This should be set to the number of CPUs
    available to the python script."""
    sequential_local_fold_fitting: bool = False
    """Use Ray for local fold fitting. This is used to speed up the local fold fitting and force
    this behavior if True. If False the default strategy of running the local fold fitting is used,
    as determined by AutoGluon and the model's default_ag_args_ensemble parameters. Should only be used for
    debugging anymore."""
    model_artifacts_base_path: str | Path | None = "/tmp/ag"  # noqa: S108
    """Adapt the default temporary directory used for model artifacts in TabArena.
        - If None, the default temporary directory is used: "./AutoGluonModels".
        - If a string or Path, the directory is used as the base path for the temporary
        and any model artifacts will be stored in time-stamped subdirectories.
    """

    @property
    def slurm_job_json(self) -> str:
        """JSON file with the job data to run used by SLURM. This is generated from the configs and metadata."""
        return f"slurm_run_data_{self.benchmark_name}.json"

    @property
    def configs(self) -> str:
        """YAML file with the configs to run. Generated from parameters above in code below."""
        return f"{self.base_path}{self.configs_path_from_base_path}{self.benchmark_name}.yaml"

    @property
    def output_dir(self) -> str:
        """Output directory for the benchmark."""
        return (
            self.base_path + self.output_dir_base_from_base_path + self.benchmark_name
        )

    @property
    def metadata(self) -> str:
        """Dataset/task Metadata for TabArena."""
        return self.base_path + self.metadata_from_base_path

    @property
    def python(self) -> str:
        """Python executable to use for the SLURM jobs."""
        return self.base_path + self.python_from_base_path

    @property
    def run_script(self) -> str:
        """Python script to run the benchmark."""
        return self.base_path + self.run_script_from_base_path

    @property
    def openml_cache(self) -> str:
        """OpenML cache directory."""
        return self.base_path + self.openml_cache_from_base_path

    @property
    def tabrepo_cache_dir(self) -> str:
        """TabRepo cache directory."""
        return self.base_path + self.tabrepo_cache_dir_from_base_path

    @property
    def slurm_log_output(self) -> str:
        """Directory for the SLURM output logs."""
        return self.base_path + self.slurm_log_output_from_base_path

    @property
    def slurm_script(self) -> str:
        """SLURM script to run the benchmark."""
        script = self.slurm_script_gpu if self.num_gpus > 0 else self.slurm_script_cpu
        return str(Path(__file__).parent / script)

    def get_jobs_to_run(self):  # noqa: C901
        """Determine all jobs to run by checking the cache and filtering
        invalid jobs.
        """
        Path(self.openml_cache).mkdir(parents=True, exist_ok=True)
        Path(self.tabrepo_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.slurm_log_output).mkdir(parents=True, exist_ok=True)

        if self.custom_metadata is not None:
            metadata = deepcopy(self.custom_metadata)
        else:
            metadata = pd.read_csv(self.metadata)

        self.generate_configs_yaml()
        # Read YAML file and get the number of configs
        with open(self.configs) as file:
            configs = yaml.safe_load(file)["methods"]

        def yield_all_jobs():
            for row in metadata.itertuples():
                repeats_folds = product(
                    range(int(row.tabarena_num_repeats)), range(int(row.num_folds))
                )
                if self.tabarena_lite:  # Filter to only first split.
                    repeats_folds = list(repeats_folds)[:1]

                for repeat_i, fold_i in repeats_folds:
                    for config_index, config in list(enumerate(configs)):
                        task_id = row.task_id
                        can_run_tabicl = row.can_run_tabicl
                        can_run_tabpfnv2 = row.can_run_tabpfnv2

                        # Quick, model independent skip.
                        if row.problem_type not in self.problem_types_to_run:
                            continue

                        yield {
                            "config_index": config_index,
                            "config": config,
                            "task_id": task_id,
                            "fold_i": fold_i,
                            "repeat_i": repeat_i,
                            "can_run_tabicl": can_run_tabicl,
                            "can_run_tabpfnv2": can_run_tabpfnv2,
                        }

        jobs_to_check = list(yield_all_jobs())

        # Check cache and filter invalid jobs in parallel using Ray
        if ray.is_initialized:
            ray.shutdown()
        ray.init(num_cpus=self.num_ray_cpus)
        output = ray_map_list(
            list_to_map=list(to_batch_list(jobs_to_check, 10_000)),
            func=should_run_job_batch,
            func_element_key_string="input_data_list",
            num_workers=self.num_ray_cpus,
            num_cpus_per_worker=1,
            func_kwargs={
                "output_dir": self.output_dir,
                "cache_path_format": self.cache_path_format,
                "cache_cls": self.cache_cls,
                "cache_cls_kwargs": self.cache_cls_kwargs,
            },
            track_progress=True,
            tqdm_kwargs={"desc": "Checking Cache and Filter Invalid Jobs"},
        )
        output = [
            item for sublist in output for item in sublist
        ]  # Flatten the batched list
        to_run_job_map = {}
        for run_job, job_data in zip(output, jobs_to_check, strict=True):
            if run_job:
                job_key = (
                    job_data["task_id"],
                    job_data["fold_i"],
                    job_data["repeat_i"],
                )
                if job_key not in to_run_job_map:
                    to_run_job_map[job_key] = []
                to_run_job_map[job_key].append(job_data["config_index"])

        # Convert the map to a list of jobs
        jobs = []
        to_run_jobs = 0
        for job_key, config_indices in to_run_job_map.items():
            to_run_jobs += len(config_indices)
            for config_batch in to_batch_list(config_indices, self.methods_per_job):
                jobs.append(
                    {
                        "task_id": job_key[0],
                        "fold": job_key[1],
                        "repeat": job_key[2],
                        "config_index": config_batch,
                    },
                )

        print(f"Generated {to_run_jobs} jobs to run without batching.")
        print(f"Jobs with batching: {len(jobs)}")
        return jobs

    def generate_configs_yaml(self):
        """Generate the YAML file with the configurations to run based
        on specific models to run.
        """
        from tabrepo.benchmark.experiment import (
            AGModelBagExperiment,
            YamlExperimentSerializer,
        )
        from tabrepo.models.utils import get_configs_generator_from_name

        experiments_lst = []
        method_kwargs = {}
        if self.model_artifacts_base_path is not None:
            method_kwargs["init_kwargs"] = {
                "default_base_path": self.model_artifacts_base_path
            }

        print(
            "Generating experiments for models...",
            f"\n\t`all` := number of configs: {self.n_random_configs}",
            f"\n\t{len(self.models)} models: {self.models}",
            f"\n\t{len(self.preprocessing_pieplines)} preprocessing pipelines: {self.preprocessing_pieplines}",
            f"\n\tMethod kwargs: {method_kwargs}",
        )
        for preprocessing_name in self.preprocessing_pieplines:
            pipeline_method_kwargs = deepcopy(method_kwargs)

            name_id_suffix = ""
            if preprocessing_name != "default":
                pipeline_method_kwargs["preprocessing_pipeline"] = preprocessing_name
                name_id_suffix = f"_{preprocessing_name}"

            for model_name, n_configs, seed_config in self.models:
                if isinstance(n_configs, str) and n_configs == "all":
                    n_configs = self.n_random_configs
                elif not isinstance(n_configs, int):
                    raise ValueError(
                        f"Invalid number of configurations for model {model_name}: {n_configs}. "
                        "Must be an integer or 'all'."
                    )
                config_generator = get_configs_generator_from_name(model_name)
                experiments_lst.append(
                    config_generator.generate_all_bag_experiments(
                        num_random_configs=n_configs,
                        add_seed=seed_config,
                        name_id_suffix=name_id_suffix,
                        method_kwargs=pipeline_method_kwargs,
                    )
                )

        # Post Process experiment list
        experiments_all: list[AGModelBagExperiment] = [
            exp for exp_family_lst in experiments_lst for exp in exp_family_lst
        ]

        # Verify no duplicate names
        experiment_names = set()
        for experiment in experiments_all:
            if experiment.name not in experiment_names:
                experiment_names.add(experiment.name)
            else:
                raise AssertionError(
                    f"Found multiple instances of experiment named {experiment.name}. All experiment names must be unique!",
                )

        YamlExperimentSerializer.to_yaml(experiments=experiments_all, path=self.configs)

    def get_jobs_dict(self):
        """Get the jobs to run as a dictionary with default arguments and jobs."""
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

    def setup_jobs(self):
        """Setup the jobs to run by generating the SLURM job JSON file."""
        jobs_dict = self.get_jobs_dict()
        n_jobs = len(jobs_dict["jobs"])
        if n_jobs == 0:
            print("No jobs to run.")
            Path(self.slurm_job_json).unlink(missing_ok=True)
            Path(self.configs).unlink(missing_ok=True)
            return

        with open(self.slurm_job_json, "w") as f:
            json.dump(jobs_dict, f)

        print(
            f"##### Setup Jobs for {self.benchmark_name}"
            "\nRun the following command to start the jobs:"
            f"\nsbatch --array=0-{n_jobs - 1}%100 {self.slurm_script} {self.slurm_job_json}"
            "\n"
        )

    @staticmethod
    def models_for_tabpfnv2_subset() -> list[str]:
        """Models within TabPFNv2 constraints.

        - <=10k training samples
        - <=500 features
        - <=10 classes
        """
        return ["TA-TABPFNV2", "TABPFNV2", "MITRA"]

    @staticmethod
    def models_for_tabicl_subset() -> list[str]:
        """Models within TabICL constraints.

        - <=10k training samples
        - <=500 features
        """
        return ["TA-TABICL", "TABICL"]


def should_run_job_batch(*, input_data_list: list[dict], **kwargs) -> list[bool]:
    """Batched version for Ray."""
    return [should_run_job(input_data=data, **kwargs) for data in input_data_list]


def should_run_job(
    *,
    input_data: dict,
    output_dir: str,
    cache_path_format: str,
    cache_cls: CacheFunctionPickle,
    cache_cls_kwargs: dict,
) -> bool:
    """Check if a job should be run based on the configuration and cache.
    Must be not a class function to be used with Ray.
    """
    config = input_data["config"]
    task_id = input_data["task_id"]
    fold_i = input_data["fold_i"]
    repeat_i = input_data["repeat_i"]
    can_run_tabicl = input_data["can_run_tabicl"]
    can_run_tabpfnv2 = input_data["can_run_tabpfnv2"]

    # Check if local task or not
    try:
        task_id = int(task_id)
    except ValueError:
        task_id = task_id.split("|", 2)[
            1
        ]  # Extract the local task ID if it is a UserTask.task_id_str

    # Filter out-of-constraints datasets
    if (
        # Skip TabICL if the dataset cannot run it
        (
            (config["model_cls"] in BenchmarkSetup.models_for_tabicl_subset())
            and (not can_run_tabicl)
        )
        # Skip TabPFN if the dataset cannot run it
        or (
            (config["model_cls"] in BenchmarkSetup.models_for_tabpfnv2_subset())
            and (not can_run_tabpfnv2)
        )
    ):
        # Reset total_jobs and update progress bar
        return False

    return not check_cache_hit(
        result_dir=output_dir,
        method_name=config["name"],
        task_id=task_id,
        fold=fold_i,
        repeat=repeat_i,
        cache_path_format=cache_path_format,
        cache_cls=cache_cls,
        cache_cls_kwargs=cache_cls_kwargs,
        mode="local",
    )
