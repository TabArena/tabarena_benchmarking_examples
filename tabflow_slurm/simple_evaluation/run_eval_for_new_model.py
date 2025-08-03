from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelMetadata:
    """Metadata related to the result artifacts for a custom model to be evaluated on TabArena."""

    path_raw: Path
    """Path to the directory containing raw results from the custom model.
    If None, defaults to a predefined path."""
    method: str
    """Name of the custom method to be evaluated. This is the `ag_name` key in the method class."""
    new_result_prefix: str | None = None
    """Optional prefix for the new results. If None, defaults to the method name."""
    only_load_cache: bool = False
    """If False, the results will be computed and cached. If True, only loads the cache."""


def run_eval_for_new_models(
    models: list[ModelMetadata],
    *,
    fig_output_dir: Path,
    subset: str | None | list[str] = None,
    cache_path: str | None = None,
) -> None:
    """Run evaluation for a custom model on TabArena.

    Args:
        fig_output_dir: Path to the directory where evaluation artifacts will be saved.
        subset: Optional subset of the TabArena dataset to evaluate on.
        cache: If True, caches the results and metadata to avoid recomputation.
        cache_path: Optional path to the cache directory on the filesystem.

    """
    if cache_path is not None:
        import os

        os.environ["TABARENA_CACHE"] = cache_path

    from tabrepo.nips2025_utils.end_to_end import EndToEndResults, compare_on_tabarena

    model_results = []
    for model in models:
        if not model.only_load_cache:
            try:
                from tabrepo.nips2025_utils.end_to_end import (
                    create_and_cache_end_to_end_results,
                )

                create_and_cache_end_to_end_results(
                    path_raw=model.path_raw / "data",
                    num_cpus=32,
                    artifact_name=model.new_result_prefix,
                )
            except:
                from tabrepo.nips2025_utils.end_to_end import EndToEnd, EndToEndResults

                _ = EndToEnd.from_path_raw(path_raw=model.path_raw / "data")

        # Load the results needed for comparison.
        end_to_end_results = EndToEndResults.from_cache(
            method=model.method, artifact_name=model.new_result_prefix
        )
        model_results.append(
            end_to_end_results.get_data_to_compare_on_tabarena(
                new_result_prefix=model.new_result_prefix,
            )
        )

    # Compare to precomputed results
    leaderboard = compare_on_tabarena(
        new_results=model_results,
        output_dir=fig_output_dir,
        subset=subset,
    )
    print(leaderboard)


if __name__ == "__main__":
    fig_dir = Path(__file__).parent / "evals"
    out_dir = Path("/work/dlclarge2/purucker-tabarena/output")

    # Eval for model seed experiment
    run_eval_for_new_models(
        [
            ModelMetadata(
                path_raw=out_dir / "realmlp_0108_seed_static",
                method="RealMLP",
                new_result_prefix="STATIC",
            ),
            ModelMetadata(
                path_raw=out_dir / "realmlp_0108_seed_fw",
                method="RealMLP",
                new_result_prefix="FOLD-WISE",
            ),
            ModelMetadata(
                path_raw=out_dir / "realmlp_0108_seed_fcw",
                method="RealMLP",
                new_result_prefix="FOLD-CONFIG-WISE",
            ),
        ],
        fig_output_dir=fig_dir / "model_seed_experiment",
        cache_path="/work/dlclarge2/purucker-tabarena/output/tabarena_cache",
        subset=["lite"],
    )


    # --- Old eval
    # for result_data in [
    # {
    #     "method": "BetaTabPFN",
    #     "path_raw": "/work/dlclarge2/purucker-tabarena/output/beta_tabpfn_rerun/data",
    #     "fig_output_dir": base_output_dir / "beta_tabpfn",
    #     "subset": "classification",
    #     "cache": False,
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
    # {
    #     "method": "ExplainableBM",
    #     "path_raw": "/work/dlclarge2/purucker-tabarena/output/ebm_30062025/data",
    #     "fig_output_dir": base_output_dir / "ebm_30062025",
    #     "new_result_prefix": "[NEW]",
    #     "cache": True,
    #     "subset": "classification",
    # },
    # {
    #     "method": "boosted_dpdt",
    #     "path_raw": "/work/dlclarge2/purucker-tabarena/output/bdpdt/data",
    #     "fig_output_dir": base_output_dir / "bdpdt",
    #     "subset": ["classification", "lite"],
    #     "cache": True,
    # },
    # ]:
    #     run_eval_for_new_model(
    #         **result_data,
    #         cache_path="/work/dlclarge2/purucker-tabarena/output/tabarena_cache",
    #     )
