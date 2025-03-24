from ppm_benchmark.models.base_dataset_normalizer import BaseDatasetNormalizer
import pandas as pd


class BPI2014Normalizer(BaseDatasetNormalizer):

    def __init__(self):
        super().__init__()

    def normalize_next_attribute(self, df):
        df = df.rename(columns={
            'Incident ID': 'case:concept:name',
            'DateStamp': 'time:timestamp',
        })
        df['concept:name'] = df['IncidentActivity_Number'] + '_' + df['IncidentActivity_Type']
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='%d-%m-%Y %H:%M:%S')
        return df

    def normalize_outcome(self, df):
        return self.normalize_next_attribute(df)

    def normalize_attribute_suffix(self, df):
        return self.normalize_next_attribute(df)
