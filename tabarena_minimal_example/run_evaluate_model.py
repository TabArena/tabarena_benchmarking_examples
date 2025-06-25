"""Example code to evaluate a model by comparing it to the leaderboard for TabArena(-Lite).

Before using this code, you must first run `run_tabarena_lite.py` to generate the input files.
"""

from __future__ import annotations

from pathlib import Path

from tabrepo.nips2025_utils.end_to_end import EndToEnd, EndToEndResults

if __name__ == "__main__":
    path_raw = Path(__file__).parent / "tabarena_out" / "custom_model"
    """Output directory for saving the results and result artifacts from TabArena."""

    fig_output_dir = Path(__file__).parent / "evals" / "custom_model"
    """Output for artefacts from the evaluation results of the custom model."""

    method = "CustomRF"
    cache = True

    """
    Run logic end-to-end and cache all results:
    1. load raw artifacts
        path_raw should be a directory containing `results.pkl` files for each run.
        In the current code, we require `path_raw` to contain the results of only 1 type of method.
    2. infer method_metadata
    3. cache method_metadata
    4. cache raw artifacts
    5. infer task_metadata
    5. generate processsed
    6. cache processed
    7. generate results
    8. cache results

    Once this is executed once, it does not need to be ran again.
    """
    if cache:
        end_to_end = EndToEnd.from_path_raw(path_raw=path_raw)

    """
    Load cached results and compare on TabArena
    1. Generates figures and leaderboard using the TabArena methods and the user's method
    2. Currently compares on all datasets, does not compare on subsets.
    3. Missing values are imputed to default RandomForest.
    """
    end_to_end_results = EndToEndResults.from_cache(method=method)
    leaderboard = end_to_end_results.compare_on_tabarena(output_dir=fig_output_dir)
    print(leaderboard)
