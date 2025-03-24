from ppm_benchmark.models.base_dataset_loader import BaseDatasetLoader
import requests
import zipfile
import os
import tempfile
import gzip
import shutil
import pm4py
import sys
from contextlib import redirect_stdout
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RemoteXesGz(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, url):
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = self._download_file(url, temp_dir)
            extracted_xes_file = self._extract_gz_file(filename, 'downloaded.xes')
            event_log = pm4py.read_xes(extracted_xes_file)
            df = pm4py.convert_to_dataframe(event_log)
            return df

    def _download_file(self, url, temp_dir):
        local_filename = os.path.join(temp_dir, 'downloaded.xes.gz')
        logger.info(f"Downloading eventlog from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    def _extract_gz_file(self, gz_path, extract_to):
        with gzip.open(gz_path, 'rb') as f_in:
            with open(extract_to, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return extract_to