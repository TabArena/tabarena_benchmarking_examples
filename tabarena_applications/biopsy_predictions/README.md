# Biopsy Prediction Application with TabArena

This code contains an example of using the TabArena library to benchmark SOTA ML method for biopsy 
predictions task, as required by clinicians. 

A special feature of this code is that we use TabArena with a local dataset that is not shared online.

Requires the environment from the main README.md file. For `run_on_slurm.py` also see the readme in `tabflow_slurm`.

Then, one can schedule the job on a SLURM cluster with:

```bash
# active your venv and cd to the directory of the script
source /work/dlclarge2/purucker-tabarena/venvs/tabarena_ag14/bin/activate && cd /work/dlclarge2/purucker-tabarena/code/tabarena_benchmarking_examples/tabarena_applications/biopsy_predictions
# now follow the output of run_on_slurm.py
```