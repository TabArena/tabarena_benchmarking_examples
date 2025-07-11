from __future__ import annotations

from pathlib import Path


def run_eval_for_new_model(
    *,
    path_raw: Path,
    fig_output_dir: Path,
    method: str,
    subset: str | None = None,
    cache: bool = False,
    cache_path: str | None = None,
    new_result_prefix: str | None = None,
) -> None:
    """Run evaluation for a custom model on TabArena.

    Args:
        path_raw: Path to the directory containing raw results from the custom model.
            If None, defaults to a predefined path.
        fig_output_dir: Path to the directory where evaluation artifacts will be saved.
        method: Name of the custom method to be evaluated.
        subset: Optional subset of the TabArena dataset to evaluate on.
        cache: If True, caches the results and metadata to avoid recomputation.
        cache_path: Optional path to the cache directory on the filesystem.
        new_result_prefix: Optional prefix for the new results. If None, defaults to
            the method name.
    """
    if cache_path is not None:
        import os

        os.environ["TABARENA_CACHE"] = cache_path

    from tabrepo.nips2025_utils.end_to_end import create_and_cache_end_to_end_results, EndToEndResults

    if cache:
        create_and_cache_end_to_end_results(path_raw=path_raw)
    end_to_end_results = EndToEndResults.from_cache(method=method)
    leaderboard = end_to_end_results.compare_on_tabarena(
        output_dir=fig_output_dir, subset=subset, new_result_prefix=new_result_prefix,
    )
    print(leaderboard)


if __name__ == "__main__":
    base_output_dir = Path(__file__).parent / "evals"
    for result_data in [
        # {
        #     "method": "BetaTabPFN",
        #     "path_raw": "/work/dlclarge2/purucker-tabarena/output/beta_tabpfn/data",
        #     "fig_output_dir": base_output_dir / "beta_tabpfn",
        #     "subset": "classification",
        # },
        # {
        #     "method": "TabFlex",
        #     "path_raw": "/work/dlclarge2/purucker-tabarena/output/tabflex/data",
        #     "fig_output_dir": base_output_dir / "tabflex",
        #     "subset": "classification",
        # },
        # {
        #     "method": "PerpetualBoosting",
        #     "path_raw": "/work/dlclarge2/purucker-tabarena/output/perpetual/data",
        #     "fig_output_dir": base_output_dir / "perpetual_boosting",
        # },
        # {
        #     "method": "TabICL",
        #     "path_raw": "/work/dlclarge2/purucker-tabarena/output/tabicl/data",
        #     "fig_output_dir": base_output_dir / "tabicl",
        #     "subset": "TabICL",
        # },
        {
            "method": "ExplainableBM",
            "path_raw": "/work/dlclarge2/purucker-tabarena/output/ebm_30062025/data",
            "fig_output_dir": base_output_dir / "ebm_30062025",
            "new_result_prefix": "[NEW]",
            "cache": True,
            "subset": "classification",
        },
    ]:
        run_eval_for_new_model(
            **result_data,
            cache_path="/work/dlclarge2/purucker-tabarena/output/tabarena_cache",
        )
