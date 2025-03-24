from ppm_benchmark.models.base_metric import BaseMetric
from sklearn.metrics import mean_squared_error


class MSE(BaseMetric):

    def __init__(self):
        super().__init__('MSE')

    def evaluate(self, predictions, targets):
        return mean_squared_error(list(targets), list(predictions))
