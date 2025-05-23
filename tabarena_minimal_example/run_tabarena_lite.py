"""Example code to run TabArena(-Lite) experiments with a custom model."""

from __future__ import annotations

from pathlib import Path

import openml
from custom_tabarena_model import get_configs_for_custom_rf
from tabrepo.benchmark.experiment import run_experiments


def run_tabarena_lite_for_custom_rf(output_dir: str):
    """Put all the code together to run a TabArenaLite experiment for
    the custom random forest model.

    Parameters
    ----------
    output_dir : str
        The path to the output directory where the results will be saved (and cached).
    """
    # Get all tasks from TabArena-v0.1
    task_ids = openml.study.get_suite("tabarena-v0.1").tasks
    # This might take a while, as it initializes the cache and downloads the datasets.
    task_metadata = {
        task_id: openml.tasks.get_task(task_id).get_dataset().name
        for task_id in task_ids
    }

    # Gets 1 default and 1 random config
    methods = get_configs_for_custom_rf(default_config=True, num_random_configs=1)

    run_experiments(
        expname=output_dir,
        tids=task_ids,
        task_metadata=task_metadata,
        methods=methods,
        ignore_cache=False,  # If True, rerun and overwrite existing results.
        # TabArena-Lite only runs on the first split of each dataset.
        repeat_fold_pairs=[(0, 0)],
        folds=None,
        repeats=None,
        # Other args
        cache_cls_kwargs={"include_self_in_call": True},
        debug_mode=False,
    )


if __name__ == "__main__":
    run_tabarena_lite_for_custom_rf(
        output_dir=str(Path(__file__).parent / "tabarena_out"),
    )
