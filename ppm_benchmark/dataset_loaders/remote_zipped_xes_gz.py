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


class RemoteZippedXesGz(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, url):
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = self._download_and_extract_zip(url, temp_dir)

            xes_gz_file = None
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith('.xes.gz'):
                        xes_gz_file = os.path.join(root, file)
                        break

            if xes_gz_file is None:
                raise FileNotFoundError("No .xes.gz file found in the extracted zip")

            xes_file = os.path.join(temp_dir, 'extracted.xes')
            extracted_xes_file = self._extract_gz_file(xes_gz_file, xes_file)
            event_log = pm4py.read_xes(extracted_xes_file)
            df = pm4py.convert_to_dataframe(event_log)
            return df

    def _download_and_extract_zip(self, url, temp_dir):
        local_filename = os.path.join(temp_dir, 'downloaded.zip')
        logger.info(f"Downloading eventlog from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        return temp_dir

    def _extract_gz_file(self, gz_path, extract_to):
        with gzip.open(gz_path, 'rb') as f_in:
            with open(extract_to, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return extract_to