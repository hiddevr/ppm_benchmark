from ppm_benchmark.models.base_dataset_loader import BaseDatasetLoader
import tempfile
import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


class BPI2016LocalLoader(BaseDatasetLoader):

    def __init__(self):
        super().__init__()

    def load_data(self, url):
        dfs = {}
        file_paths = {
            'BPI2016_Clicks_Logged_In': '../../raw_eventlogs/BPI2016_Clicks_Logged_In.csv/BPI2016_Clicks_Logged_In.csv',
            'BPI2016_Complaints': '../../raw_eventlogs/BPI2016_Complaints.csv',
            'BPI2016_Questions': '../../raw_eventlogs/BPI2016_Questions.csv',
            'BPI2016_Werkmap_Messages': '../../raw_eventlogs/BPI2016_Werkmap_Messages.csv'
        }
        for name, file_path in file_paths.items():
            event_log = pd.read_csv(file_path, sep=';', encoding='latin-1')
            dfs[name] = event_log
        return dfs
