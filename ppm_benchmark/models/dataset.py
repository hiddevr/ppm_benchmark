from .base_dataset_normalizer import BaseDatasetNormalizer
from .base_dataset_loader import BaseDatasetLoader


class Dataset:

    def __init__(self, name: str, dataset_normalizer: BaseDatasetNormalizer, dataset_loader: BaseDatasetLoader,
                 data_path: str, is_remote: bool, data_owner: str):
        self.name = name
        self.dataset_normalizer = dataset_normalizer
        self.dataset_loader = dataset_loader
        self.data_path = data_path
        self.is_remote = is_remote
        self.data_owner = data_owner
        self.raw_data = None

    def load(self):
        if self.is_remote:
            print(f'Downloading {self.name} dataset by {self.data_owner} from {self.data_path}')

        self.raw_data = self.dataset_loader.load_data(self.data_path)
        return

    def normalize_and_split(self, task_type, start_date, end_date, outcome_ltl_rule=None, attr_col=None):
        train, test = self.dataset_normalizer.normalize_and_split(self.raw_data, task_type, start_date, end_date,
                                                                  outcome_ltl_rule, attr_col)
        return train, test

    def create_task(self):
        pass
