from ppm_benchmark.models.base_dataset_loader import BaseDatasetLoader
from ppm_benchmark.utils.logger import setup_logger
import pandas as pd
import requests
import os
import tempfile


logger = setup_logger(__name__)


class RemoteCSV(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, url):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Downloading eventlog from {url}...")
            csv_file = self._download_csv(url, temp_dir)
            event_log = pd.read_csv(csv_file, delimiter=';')
            if len(event_log.columns) == 1:
                event_log = pd.read_csv(csv_file)
            return event_log

    def _download_csv(self, url, temp_dir):
        local_filename = os.path.join(temp_dir, 'downloaded.csv')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return local_filename
