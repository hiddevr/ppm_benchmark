from ppm_benchmark.models.base_dataset_normalizer import BaseDatasetNormalizer
from ..utils.logger import setup_logger
import pandas as pd


logger = setup_logger(__name__)


class HelpdeskNormalizer(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def normalize_next_attribute(self, df):
        df.rename(columns={
            'Activity': 'concept:name',
            'Case ID': 'case:concept:name',
            'Resource': 'org:resource',
            'Complete Timestamp': 'time:timestamp'
        }, inplace=True)

        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        return df

    def normalize_outcome(self, df):
        df = self.normalize_next_attribute(df)
        df['concept:name'] = df['concept:name'].str.replace(' ', '_')
        return df

    def normalize_attribute_suffix(self, df):
        return self.normalize_next_attribute(df)
