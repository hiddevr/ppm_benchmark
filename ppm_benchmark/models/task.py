import os
import pandas as pd
from .base_task_generator import BaseTaskGenerator
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Task:

    def __init__(self, name: str, task_generator: BaseTaskGenerator, save_folder: str, task_type: str):
        self.name = name
        self.task_generator = task_generator
        self.save_folder = save_folder
        self.task_type = task_type
        self.baselines = None
        self.use_proba = None

    def generate_task(self, train, test):
        evaluation_data, baselines, use_proba = self.task_generator.generate_task(train, test, self.name)
        self.baselines = baselines
        self.use_proba = use_proba

        logger.debug(f'Baselines after task generation {self.baselines}')

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        test.drop(columns=['fraction_complete'], inplace=True)
        if 'outcome_satisfied' in test.columns:
            train.drop(columns=['outcome_satisfied'], inplace=True)
            test.drop(columns=['outcome_satisfied'], inplace=True)

        train.to_csv(os.path.join(self.save_folder, 'train.csv'), index=None)
        test.to_csv(os.path.join(self.save_folder, 'test.csv'), index=None)

        return evaluation_data

    def get_train_data(self):
        train_df = pd.read_csv(os.path.join(self.save_folder, 'train.csv'), low_memory=False)
        return train_df

    def get_test_data(self):
        test_df = pd.read_csv(os.path.join(self.save_folder, 'test.csv'), low_memory=False)
        return test_df
