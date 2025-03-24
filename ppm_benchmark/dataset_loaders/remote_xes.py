from ppm_benchmark.models.base_dataset_loader import BaseDatasetLoader
import tempfile
import os
import requests
import pm4py
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class RemoteXes(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, url):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Downloading eventlog from {url}...")
            xes_file = self._download_xes(url, temp_dir)
            event_log = pm4py.read_xes(xes_file)
            df = pm4py.convert_to_dataframe(event_log)
            return df

    def _download_xes(self, url, temp_dir):
        local_filename = os.path.join(temp_dir, 'downloaded.xes')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return local_filename
