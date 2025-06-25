# TabArena Benchmarking Examples

This repository contains examples for benchmarking predictive machine learning models with TabArena.

## Content Overview

* `./tabarena_minimal_example/` - contains a minimal example of how to use TabArena for benchmarking.
* `./tabflow_slurm/` - contains code for benchmarking with TabArena on a SLURM cluster.

## Download Benchmarking Examples Repo

```bash
git clone https://github.com/TabArena/tabarena_benchmarking_examples.git
cd tabarena_benchmarking_examples
```

## Install Benchmarking Environment

We recommend to use `uv` and Python 3.11 and a Linux OS. The tutorial below already integrates this into the installation process.

```bash
pip install uv
uv venv --seed --python 3.11 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate

# get editable external libraries
cd external_libs
git clone --branch main https://github.com/autogluon/tabrepo.git

# use GIT_LFS_SKIP_SMUDGE=1 in front of the command if installing TabDPT fails due to a broken LFS/pip setup
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e tabrepo/[benchmark]

# When planning to only run experiments on CPU, also run the following:
uv pip install -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

For PyCharm users, remember to mark `tabrepo` under `external_libs` as Source Roots (right click
-> Mark Directory as -> Source Root).

Test your installation via the code below. This might take some time to download the foundation models, see `tabflow_slurm/benchmarking_setup/download_all_foundation_models.py` to download all models beforehand if needed.
```bash
pytest external_libs/tabrepo/tst/benchmark/models/
```
