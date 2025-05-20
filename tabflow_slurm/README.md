# TabArena Benchmarking on SLURM (TabFlow Slurm)

This directory contains our code to run TabArena on SLURM. While still work-in-progress, we used this code to run all
GPU-related experiments. We detail the instructions and insights below.

We expect this code to be helpful but not work out of the box for any SLURM cluster as the nature of SLUR deployment
in the wild is extremely heterogeneous. We provide a few instructions and insights below to help you adapt the
code to your needs. If you find any problems, please open an issue. We aim to make the code as generally applicable
as possible in the future.

## Code Overview

* `benchmarking_setup/` - contains the code to set up benchmarking on SLURM (see its README for more details).
* `models/` - contains the code to generate model configs that are to be benchmarked (see its README for more details).
* `run_tabarena_experiment.py` - contains code to run an individual experiment. This is the code that is run in one jobs
  on the SLURM cluster.
* `run_setup_slurm_jobs.py` - contains code to set up jobs we want to submit to the SLURM cluster (checking duplicates,
  which jobs have to be run, etc.).
* `submit_template_gpu.sh` - contains the array job template for our SLURM jobs and is called/executed after running
  `run_setup_slurm_jobs.py`.

## Usage Example

1. Make sure to run all code in `./benchmarking_setup/`
2. Generate the configs you want to benchmark in `./models/`.
3. Adapt all options in `submit_template_gpu.sh` and `run_setup_slurm_jobs.py` for your local SLURM setup (partitions,
   paths, etc.).
4. In a CLI on a (login) node of the SLURM cluster, active your virtual environment and navigate to the correct
   directory, for example:

```bash
source /work/dlclarge2/purucker-tabarena/venvs/tabarena_gpu/bin/activate && cd /work/dlclarge2/purucker-tabarena/code/tabarena/tabflow_slurm
```

5. Run `run_setup_slurm_jobs.py` to set up all data needed to run array jobs and submit the array job to the slurm
   cluster by following the printed instructions. 
