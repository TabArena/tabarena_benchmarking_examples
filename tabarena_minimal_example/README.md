# TabArena-Lite Minimal Example

This is a minimal example of how to use TabArena for benchmarking. 
It includes the following files:

## Using TabArena for Benchmarking

- `custom_tabarena_model.py` - Example for implementing a scikit-learn-API-compatible model (RandomForest) for TabArena.
- `run_tabarena_lite.py` - Example for running TabArena-Lite with the custom model.
- `run_evaluate_model.py` - Example for evaluating the custom model with TabArena-Lite vs. the leaderboard.
- `run_tabarena_on_custom_dataset.py` - Example for benchmarking TabArena models on a custom dataset (without OpenML).

## Running TabArena Models 
- `run_tabarena_model.py` - Minimal example for running a model from TabArena on a new dataset (without the benchmarking code of TabArena).
- `running_tabarena_models/` - Extended examples for running TabArena models

## Using the Data and Tasks from TabArena
- `get_tabarena_data.py` - Example for downloading TabArena data and tasks from OpenML.