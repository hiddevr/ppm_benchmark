from ppm_benchmark.models.base_dataset_normalizer import BaseDatasetNormalizer


class BPI2015Normalizer(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def normalize_next_attribute(self, df):
        return df.drop(['dateFinished'], axis=1)

    def normalize_outcome(self, df):
        return self.normalize_next_attribute(df)

    def normalize_attribute_suffix(self, df):
        return self.normalize_next_attribute(df)
