"""Example code to run TabArena(-Lite) experiments with a custom model."""

from __future__ import annotations

from pathlib import Path

from tabrepo import EvaluationRepository
from tabrepo.nips2025_utils.fetch_metadata import load_task_metadata
from tabrepo.nips2025_utils.generate_repo import generate_repo as _generate_repo


def generate_repo(input_dir: str):
    """
    Must first run `run_tabarena_lite.py` to generate the input files.

    Parameters
    ----------
    input_dir : str
        The folder location of results, need to point this to the correct folder
    """

    # Load Context
    # expname = f"{tabarena_data_root}/{experiment_name}"  # folder location of results, need to point this to the correct folder
    repo_dir = "repos/ExampleRepo"  # location of local cache for fast script running

    task_metadata = load_task_metadata(paper=True)

    repo: EvaluationRepository = _generate_repo(experiment_path=input_dir, task_metadata=task_metadata)
    repo.to_dir(repo_dir)


if __name__ == "__main__":
    generate_repo(
        input_dir=str(Path(__file__).parent / "tabarena_out"),
    )
