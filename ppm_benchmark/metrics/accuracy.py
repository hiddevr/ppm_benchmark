from ppm_benchmark.models.base_metric import BaseMetric
from sklearn.metrics import accuracy_score


class Accuracy(BaseMetric):

    def __init__(self):
        super().__init__('Accuracy')

    def _predictions_from_probas(self, prediction_probas):
        preds = [max(d, key=d.get) for d in prediction_probas]
        print(preds)
        return preds

    def evaluate(self, predictions, targets):
        predictions = self._predictions_from_probas(predictions)
        return accuracy_score(list(targets), list(predictions))
