from ppm_benchmark.models.base_dataset_loader import BaseDatasetLoader
import pandas as pd


class LocalCSV(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, file_path):
        event_log = pd.read_csv(file_path, delimiter=';')
        if len(event_log.columns) == 1:
            event_log = pd.read_csv(file_path)
        return event_log
