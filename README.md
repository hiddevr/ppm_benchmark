# PPM Benchmark
A package that provides utilities for generating reliable benchmark datasets for Predictive Process Monitoring models. Based on the debiasing ideas proposed [here](https://github.com/hansweytjens/predictive-process-monitoring-benchmarks).

## Installation
It is recommended to use this package with **Python 3.8**. This is because this package depends on [fastDamerauLevenshtein](https://pypi.org/project/fastDamerauLevenshtein/) which I could not get to work on higher python versions. See the following files if you would like to implement your own version:
- [damerau_levenshtein.py](ppm_benchmark/metrics/damerau_levenshtein.py)
- [sequence_matcher.py](ppm_benchmark/utils/sequence_matcher.py)

Run the following command from the project root to install this package and its dependencies:
```shell
pip install .
```

## Usage Examples
See the [examples](examples) folder for a complete set of example notebooks.

## Configuration files
This package works using [YAML configuration files](benchmark_configs). In these files, you can specify which datasets are used, their files paths (or remote download url), evaluation metrics, and the prediction task. The following prediction tasks are supported:
- Next attribute classification (i.e. next activity prediction)
- Next attribute regression* (i.e. next timestamp prediction)
- Outcome classification*
- Outcome regression (i.e. remaining time prediction)
- Attribute suffix* classification (i.e. activity suffix prediction)

\* These have not been thoroughly tested. You might need to make some small bug fixes.

## Generating tasks
After creating a configuration file, you can run the code below to generate benchmark tasks. This will automatically save all generated tasks/datasets in the current directory.
```python
from ppm_benchmark.core.benchmark_loader import BenchmarkLoader


loader = BenchmarkLoader()
benchmark = loader.load_from_config('../benchmark_configs/remote/remote_next_attribute_classification.yaml', max_workers=2)

tasks = benchmark.get_tasks()
```

You can then load the generated tasks using:

```python
from ppm_benchmark.core.benchmark_loader import BenchmarkLoader

loader = BenchmarkLoader()
benchmark = loader.load_from_folder('next_attribute_classification')
task_names = benchmark.get_tasks()
```

Then load a task using its name:
```python
task = benchmark.load_task(task_names[0])
train = task.get_train_data()
```

## Evaluation
For classification tasks, you need to submit your predictions to the Evaluator in a list of dict with the labels as keys and predicted probabilities as values:
```python
result = [
    {"class_A": 0.80, "class_B": 0.20},
    {"class_A": 0.75, "class_B": 0.25},
    ...
]
```
For regression tasks, simply pass a list of predictions.

Then add your predictions to the evaluator as follows:
```python
from ppm_benchmark.core.benchmark_loader import BenchmarkLoader


evaluator = benchmark.get_evaluator()
evaluator.add_predictions(task_name, result, 'RF', None)
```
The None argument given here corresponds to the  'drop_type' argument. This was used for some experiments. You will likely not need it if training your models from scratch on this data. It allows you to drop certain rows from the test data if those cannot be predicted by your model.

You can add multiple models by simply calling the add_predictions() method multiple times.

### Raw metrics
Calling this will output a dataframe with the scores for all models on all metrics specified in the configuration file:
```python
evaluator.evaluate()
```

### Plots
The following methods can be used to generate plots:
```python
# Set single_row to True if using less than 3 datasets. 
# This example uses 'Accuracy' as the metric, but you can use any metric specified in the config file.
# Set use_pgf to True if you want to output plots as PGF files (make sure you have the appropriate LateX packages installed).

evaluator.plot_by_train_distance('Accuracy', 'next_attribute_classification', save=True, use_pgf=False, single_row=True)

evaluator.plot_attr_drift('Accuracy', 'next_attribute_classification', save=True, use_pgf=False, single_row=True)

evaluator.plot_by_fraction_completed('Accuracy', 'next_attribute_classification', save=True, use_pgf=False, single_row=True)
```
