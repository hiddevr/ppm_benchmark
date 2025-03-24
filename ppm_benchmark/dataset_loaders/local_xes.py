from ppm_benchmark.models.base_dataset_loader import BaseDatasetLoader
import pm4py


class LocalXes(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, file_path):
        event_log = pm4py.read_xes(file_path)
        return event_log
