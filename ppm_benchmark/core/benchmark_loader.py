import yaml
import os
from ppm_benchmark.models.benchmark import Benchmark
from ppm_benchmark.models.dataset import Dataset
from ppm_benchmark.models.task import Task
from ppm_benchmark.evaluation.evaluator import Evaluator
import importlib
import pickle
from collections import defaultdict
from ..utils.logger import setup_logger
from concurrent.futures import ProcessPoolExecutor, as_completed


logger = setup_logger(__name__)


class BenchmarkLoader:

    def __init__(self):
        pass

    def _load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _get_datasets(self, config):
        normalizer_module = importlib.import_module('ppm_benchmark.dataset_normalizers')
        loader_module = importlib.import_module('ppm_benchmark.dataset_loaders')
        datasets = []
        for dataset in config['datasets']:
            normalizer = getattr(normalizer_module, dataset['dataset_normalizer'])
            loader = getattr(loader_module, dataset['dataset_loader'])
            data_path = dataset['data_path']
            is_remote = dataset['is_remote']
            data_owner = dataset['data_owner']
            name = dataset['name']
            dataset_obj = Dataset(name, normalizer(), loader(), data_path, is_remote, data_owner)
            datasets.append(dataset_obj)
        return datasets

    def _get_metrics(self, config):
        module = importlib.import_module('ppm_benchmark.metrics')
        metrics = []
        for metric in config['benchmark']['metrics']:
            cls = getattr(module, metric['name'])
            metrics.append(cls())
        return metrics

    def _get_task_generator(self, task):
        module = importlib.import_module('ppm_benchmark.task_generators')
        task_generator = task['task_generator']
        cls = getattr(module, task_generator['name'])
        return cls()

    def _get_tasks(self, config):
        tasks = config['benchmark']['tasks']
        final_tasks = []
        for task in tasks:
            task_generator = self._get_task_generator(task)
            task_obj = Task(task['name'], task_generator, task['save_folder'], config['benchmark']['task_type'])
            final_tasks.append(task_obj)
        return final_tasks

    def _init_tasks(self, tasks, datasets, config, max_workers=1):
        normalized_datasets = defaultdict(dict)
        evaluation_data = []
        updated_tasks = []  # To hold updated tasks from parallel execution

        # Parallelize dataset normalization
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_dataset = {executor.submit(self._normalize_dataset, dataset,
                                                 config['benchmark']['task_type'],
                                                 dataset_config['split_details']['start_date'],
                                                 dataset_config['split_details']['end_date'],
                                                 dataset_config.get('ltl_rule', {}).get('rule', None),
                                                 config['benchmark']['attr_col']): dataset
                                 for dataset in datasets
                                 for dataset_config in config['datasets']
                                 if dataset_config['name'] == dataset.name}

            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    train, test = future.result()
                    normalized_datasets[dataset.name]['train'] = train
                    normalized_datasets[dataset.name]['test'] = test
                    logger.debug(f'{dataset.name} train targets: {train["target"].value_counts()}')
                    logger.debug(f'{dataset.name} test targets: {test["target"].value_counts()}')
                except Exception as exc:
                    logger.error(f'Dataset {dataset.name} generated an exception: {exc}', exc_info=True)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self._process_task, task, normalized_datasets, config): task for task in
                              tasks}

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    task_data, updated_task = future.result()
                    evaluation_data.extend(task_data)
                    updated_tasks.append(updated_task)
                    logger.info(f"Completed task: {updated_task.name}")
                except Exception as exc:
                    logger.error(f'Task {task.name} generated an exception: {exc}', exc_info=True)

        for i, updated_task in enumerate(updated_tasks):
            tasks[i] = updated_task
            logger.debug(f'Baseline updated task {tasks[i].name}: {tasks[i].baselines}')

        return evaluation_data, tasks

    def _normalize_dataset(self, dataset, task_type, start_date, end_date, outcome_ltl_rule=None, attr_col=None):
        logger.info(f"Normalizing dataset {dataset.name}...")
        dataset.load()
        return dataset.normalize_and_split(task_type, start_date, end_date, outcome_ltl_rule, attr_col)

    def _process_task(self, task, normalized_datasets, config):
        task_data = []
        for normalized_dataset in normalized_datasets.keys():
            dataset_config = None
            for dataset_in_config in config['datasets']:
                if dataset_in_config['name'] == normalized_dataset:
                    dataset_config = dataset_in_config

            dataset_task_names = [task['name'] for task in dataset_config['tasks']]
            if task.name in dataset_task_names:
                train = normalized_datasets[normalized_dataset]['train']
                test = normalized_datasets[normalized_dataset]['test']
                task_data.extend(task.generate_task(train, test))

        return task_data, task

    def load_from_config(self, config_path, max_workers=1):
        config = self._load_config(config_path)
        datasets = self._get_datasets(config)
        tasks = self._get_tasks(config)
        metrics = self._get_metrics(config)
        evaluation_data, tasks = self._init_tasks(tasks, datasets, config, max_workers)
        logger.info(f"Tasks initialized successfully.")
        logger.info(f"Initializing evaluator...")
        evaluator = Evaluator(evaluation_data, metrics, tasks[0].baselines, tasks[0].use_proba)
        logger.info(f"Evaluator initialized successfully.")
        benchmark = Benchmark(config['benchmark']['name'], tasks, evaluator)

        save_folder = config['benchmark']['save_folder']
        file_path = os.path.join(save_folder, "benchmark.pkl")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(file_path, 'wb') as file:
            pickle.dump(benchmark, file)

        return benchmark

    def load_from_folder(self, folder):
        file_path = os.path.join(folder, "benchmark.pkl")
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            return obj


