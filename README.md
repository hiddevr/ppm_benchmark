# PPM Benchmark
A package that provides utilities for generating reliable benchmark datasets for Predictive Process Monitoring models. Based on the debiasing ideas proposed [here](https://github.com/hansweytjens/predictive-process-monitoring-benchmarks).

## Installation
It is recommended to use this package with **Python 3.8**. This is because this package depends on [fastDamerauLevenshtein](https://pypi.org/project/fastDamerauLevenshtein/) which I could not get to work on higher python versions. See the following files if you would like to implement your own version:
- [damerau_levenshtein.py](ppm_benchmark/metrics/damerau_levenshtein.py)
- [sequence_matcher.py](ppm_benchmark/utils/sequence_matcher.py)

