<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <img src="https://avatars.githubusercontent.com/u/210855230" width="125" alt="TabArena Logo"/>
    </summary>
  </ul>
</div>

## Awesome TabArena Use Cases 🌻

</div>

This repository contains examples for various use cases of [TabArena](https://tabarena.ai)'s 
[code](https://github.com/autogluon/tabrepo/), a benchmarking framework for tabular data.

## 🕹️ Use Cases

### ⚡ Minimal Examples

A lightweight starting point for various TabArena use cases.

- **Folder:** `tabarena_minimal_example/`
- **Use Cases:**
    - Running the models from TabArena on your data: `running_tabarena_models/`
    - Benchmarking models with TabArena on your custom (private) dataset(s): `run_tabarena_on_custom_dataset.py`'
    - Get the data used by TabArena from OpenML, without the TabArena framework: `get_tabarena_data.py`
    - Implement your own model for TabArena and benchmark it on TabArena-Lite: `custom_tabarena_model/`


### 🔬 TabArena Applications Predictions

Example of using TabArena in a real-world application.

- **Folder:** `tabarena_applications/`
- **Highlights:**
    - Evaluating SOTA ML models on a custom (private) Biopsy dataset with TabArena: `biopsy_predictions/`


### 🖥️ TabArena on SLURM

Templates for running TabArena on SLURM clusters.

- **Folder:** `tabflow_slurm/`
- **Highlights:** All code we use to run TabArena (fully parallelized) on SLURM.

## 🪄 Install

Download Benchmarking Examples Repo

```bash
git clone https://github.com/TabArena/tabarena_benchmarking_examples.git
cd tabarena_benchmarking_examples
```

We recommend to use `uv` and Python 3.11 and a Linux OS. The tutorial below already integrates this into the
installation process.

```bash
pip install uv
uv venv --seed --python 3.11 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate

# get editable external libraries
cd external_libs
git clone --branch main https://github.com/autogluon/tabrepo.git

# Local install of AutoGluon (mostly needed for getting the latest state of the code)
git clone --branch master https://github.com/autogluon/autogluon
./autogluon/full_install.sh

# use GIT_LFS_SKIP_SMUDGE=1 in front of the command if installing TabDPT fails due to a broken LFS/pip setup
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e tabrepo/[benchmark]

# When planning to only run experiments on CPU, also run the following:
uv pip install -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

For PyCharm users, remember to mark `tabrepo` under `external_libs` as Source Roots (right click
-> Mark Directory as -> Source Root).

Test your installation via the code below. This might take some time to download the foundation models, see
`tabflow_slurm/benchmarking_setup/download_all_foundation_models.py` to download all models beforehand if needed.

```bash
pytest external_libs/tabrepo/tst/benchmark/models/
```

# 📄 Publication for TabArena

If you use TabArena in a scientific publication, we would appreciate a reference to the following paper:

**TabArena: A Living Benchmark for Machine Learning on Tabular Data**,
Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, Frank Hutter,
Preprint., 2025

Link to publication: [arXiv](https://arxiv.org/abs/2506.16791)

Bibtex entry:

```bibtex
@article{erickson2025tabarena,
  title={TabArena: A Living Benchmark for Machine Learning on Tabular Data}, 
  author={Nick Erickson and Lennart Purucker and Andrej Tschalzev and David Holzmüller and Prateek Mutalik Desai and David Salinas and Frank Hutter},
  year={2025},
  journal={arXiv preprint arXiv:2506.16791},
  url={https://arxiv.org/abs/2506.16791}, 
}
```
