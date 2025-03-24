from ppm_benchmark.models.base_dataset_normalizer import BaseDatasetNormalizer


class TrafficFineNormalizer(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def normalize_next_attribute(self, df):
        return df

    def normalize_outcome(self, df):
        df['concept:name'] = df['concept:name'].str.replace(' ', '_')
        return self.normalize_next_attribute(df)

    def normalize_attribute_suffix(self, df):
        return self.normalize_next_attribute(df)
