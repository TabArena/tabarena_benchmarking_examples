"""Example for generating configs to run in TabArena.

The code below shows how to generate configs for TabArena experiments as used in `run_tabarena_experiment.py`.
By adjusting the included model names (removing the comment) or the number of configs, you can generate configs for
different models and different numbers of configs to run.

For a new model, ideally, you can generate a similar function (e.g., see `tabrepo.models.catboost.generate`)
"""

from __future__ import annotations

from tabrepo.benchmark.experiment import AGModelBagExperiment, YamlExperimentSerializer
from tabrepo.models.utils import get_configs_generator_from_name

if __name__ == "__main__":
    n_random_configs = 200
    output_path = "configs_all_gpu.yaml"

    # Potential model names and their number of configs
    experiments_lst = []
    for model_name, n_configs in [
        ("RealMLP", 50),  # 50 for GPU here
        ("TabM", 50),  # 50 for GPU
        ("ModernNCA", 50),  # 50 for GPU
        ("TabDPT", 0),
        ("TabICL", 0),
        ("TabPFNv2", n_random_configs),
        # ("CatBoost", n_random_configs),
        # ("EBM", n_random_configs),
        # ("ExtraTrees", n_random_configs),
        # ("FastaiMLP", n_random_configs),
        # ("KNN", n_random_configs),
        # ("LightGBM", n_random_configs),
        # ("Linear", n_random_configs),
        # ("TorchMLP", n_random_configs),
        # ("RandomForest", n_random_configs),
        # ("XGBoost", n_random_configs),
    ]:
        config_generator = get_configs_generator_from_name(model_name)
        experiments_lst.append(config_generator.generate_all_bag_experiments(num_random_configs=n_configs))

    # Post Process experiment list
    experiments_all: list[AGModelBagExperiment] = [exp for exp_family_lst in experiments_lst for exp in exp_family_lst]

    # Verify no duplicate names
    experiment_names = set()
    for experiment in experiments_all:
        if experiment.name not in experiment_names:
            experiment_names.add(experiment.name)
        else:
            raise AssertionError(
                f"Found multiple instances of experiment named {experiment.name}. All experiment names must be unique!",
            )

    YamlExperimentSerializer.to_yaml(experiments=experiments_all, path=output_path)
