from ppm_benchmark.models.base_dataset_normalizer import BaseDatasetNormalizer
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BPI2020Normalizer(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def normalize_next_attribute(self, df):
        return df

    def normalize_outcome(self, df):
        df['concept:name'] = df['concept:name'].str.replace(' ', '_')
        return df

    def normalize_attribute_suffix(self, df):
        return df
