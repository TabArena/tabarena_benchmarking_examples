from __future__ import annotations

from tabicl import TabICLClassifier
from tabpfn.model.loading import download_all_models, resolve_model_path
from tabrepo.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel

if __name__ == "__main__":
    # TabPFNv2
    _, model_dir, _, _ = resolve_model_path(model_path=None, which="classifier")
    download_all_models(to=model_dir)

    # TabICL
    TabICLClassifier(checkpoint_version="tabicl-classifier-v1.1-0506.ckpt")._load_model()

    # TabDPT
    TabDPTModel._download_and_get_model_path()
