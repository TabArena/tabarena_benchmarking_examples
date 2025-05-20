# TabArena Benchmarking Examples

This repository contains examples for benchmarking predictive machine learning models with TabArena.

## Content Overview

* `./tabflow_slurm/` - contains code for benchmarking with TabArena on a SLURM cluster.

## Install Benchmarking Environment

```bash
pip install uv

# get editable external libraries
cd e_libs
git clone https://github.com/autogluon/autogluon
git clone --branch tabarena_lop https://github.com/LennartPurucker/tabrepo.git

./autogluon/full_install.sh
# use GIT_LFS_SKIP_SMUDGE=1 in front of the command if installing TabDPT fails du broken LFS/pip setup. 
uv pip install -e tabrepo/[benchmark]

# When planning to only run experiments on CPU, also run the following:
uv pip install -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

For PyCharm users, remember to mark `tabrepo` and `src` directories under `autogluon/*/src` as Source Roots (right click
-> Mark Directory as -> Source Root).
