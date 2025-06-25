"""Example code to evaluate a model by comparing it to the leaderboard for TabArena(-Lite).

Before using this code, you must first run `run_tabarena_lite.py` to generate the input files.
"""

from __future__ import annotations

from pathlib import Path

from tabrepo.nips2025_utils.end_to_end import EndToEnd, EndToEndResults


def run_eval_for_new_model(
    *, path_raw: Path, fig_output_dir: Path, method: str, cache: bool = True
) -> None:
    """Run evaluation for a custom model on TabArena.

    Args:
        path_raw: Path to the directory containing raw results from the custom model.
            If None, defaults to a predefined path.
        fig_output_dir: Path to the directory where evaluation artifacts will be saved.
        method: Name of the custom method to be evaluated.
        cache: If True, caches the results and metadata to avoid recomputation.
    """
    if cache:
        _ = EndToEnd.from_path_raw(path_raw=path_raw)
    end_to_end_results = EndToEndResults.from_cache(method=method)
    leaderboard = end_to_end_results.compare_on_tabarena(output_dir=fig_output_dir)
    print(leaderboard)


if __name__ == "__main__":
    fig_output_dir = Path(__file__).parent / "evals"
    for result_data in [
        {
            "method": "BetaTabPFN",
            "path_raw": "/work/dlclarge2/purucker-tabarena/output/beta_tabpfn/data",
            "fig_output_dir": fig_output_dir / "beta_tabpfn",
        },
        {
            "method": "TabFlex",
            "path_raw": "/work/dlclarge2/purucker-tabarena/output/tabflex/data",
            "fig_output_dir": fig_output_dir / "tabflex",
        },
        {
            "method": "PerpetualBoosting",
            "path_raw": "/work/dlclarge2/purucker-tabarena/output/perpetual/data",
            "fig_output_dir": fig_output_dir / "perpetual_boosting",
        },
    ]:
        run_eval_for_new_model(**result_data)
