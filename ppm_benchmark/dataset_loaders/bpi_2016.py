from ppm_benchmark.models.base_dataset_loader import BaseDatasetLoader
import tempfile
import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BPI2016Loader(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, url):
        dfs = {}
        urls = {
            'BPI2016_Clicks_Logged_In': 'https://data.4tu.nl/file/1c51b037-bf30-4c1a-9c50-25d1617a4899/65710417-8249-40db-a47a-dfa4d25bfad4',
            'BPI2016_Complaints': 'https://data.4tu.nl/file/0a087f36-da3a-49ec-b3d9-38a32e9617ed/46e85564-f795-4199-b179-22b9097f20fc',
            'BPI2016_Questions': 'https://data.4tu.nl/file/bfb2b689-28d4-47ce-9265-998d91b81657/63ce07c6-4693-4dac-9ad7-63488e2957ec',
            'BPI2016_Werkmap_Messages': 'https://data.4tu.nl/file/debf31e0-493e-4848-82f6-10e6630c9b0f/472e83b5-a065-41b7-8551-863a197a3028'
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, url in urls.items():
                logger.info(f"Downloading {name}...")
                csv_file = self._download_csv(url, temp_dir, name)
                event_log = pd.read_csv(csv_file, sep=';', encoding='latin-1')
                dfs[name] = event_log
        return dfs

    def _download_csv(self, url, temp_dir, name):
        local_filename = os.path.join(temp_dir, f'{name}.csv')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return local_filename
