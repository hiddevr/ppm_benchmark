from ppm_benchmark.models.base_metric import BaseMetric
import math
import statistics
import pandas as pd

class PRS(BaseMetric):

    def __init__(self):
        super().__init__('PRS')

    def normalize(self, preds, tiny_threshold=1e-12):
        new_preds = []
        for pred_dict in preds:
            # Extract scores
            scores = list(pred_dict.values())
            keys = list(pred_dict.keys())

            # Numerical stability: shift scores by max
            max_score = max(scores)
            shifted_scores = [s - max_score for s in scores]

            # Exponentiate
            exp_scores = [math.exp(s) for s in shifted_scores]
            sum_exp_scores = sum(exp_scores)

            # Softmax
            probabilities = [s / sum_exp_scores for s in exp_scores]

            # Zero out extremely tiny values
            probabilities = [
                p if p >= tiny_threshold else 0.0
                for p in probabilities
            ]

            # Re-normalize if anything got zeroed out
            sum_probs = sum(probabilities)
            if sum_probs == 0:
                # fallback strategy: if everything got zeroed out,
                # you can either distribute evenly or keep them as zero
                normalized_probs = [1.0 / len(probabilities)] * len(probabilities)
            else:
                normalized_probs = [p / sum_probs for p in probabilities]

            # Build your normalized dict
            normalized_dict = dict(zip(keys, normalized_probs))
            new_preds.append(normalized_dict)

        return pd.Series(new_preds)

    def evaluate_backup(self, predictions, targets):
        N = len(predictions)
        if N == 0:
            return None

        predictions = self.normalize(predictions)
        labels = set(predictions.iloc[0].keys()).union(set(targets.unique()))

        Rj_dict = {}
        for label in labels:
            Nj = len([pred for pred, target in zip(predictions, targets) if target == label])
            if Nj == 0:
                continue
            if Nj == N:
                Rj_dict[label] = 1
                continue

            pij_lst = [pred[label] if label in pred.keys() else 0 for pred in predictions]
            yij_lst = [1 if target == label else 0 for target in targets]

            numerator = sum(pij * (1 - yij) for pij, yij in zip(pij_lst, yij_lst)) * (1 / (N - Nj))
            denominator = sum(pij * yij for pij, yij in zip(pij_lst, yij_lst)) * (1 / Nj)
            Rj_dict[label] = numerator / denominator if denominator != 0 else 1

        return statistics.mean(Rj_dict.values())

    def evaluate(self, predictions, targets, alpha=1.0, eps=1e-7):
        """
        Compute a label-wise ratio of:
            mean(probabilities over incorrect samples)
        /  mean(probabilities over correct samples),
        then transform it with a log-sigmoid so that
        the final per-label value is in (0,1).

        :param predictions: list of dicts, each dict = {label: probability}
        :param targets: list/array of true labels (one for each prediction)
        :param labels: iterable of all possible labels
        :param alpha: controls the steepness of the log-sigmoid transform
        :param eps: small constant for numerical stability
        :return: the average of transformed ratios across all labels, in (0,1)
        """

        R_values = []
        n_samples = len(predictions)
        labels = predictions[0].keys()
        predictions = self.normalize(predictions)

        for label_j in labels:
            sum_correct_j = 0.0
            sum_incorrect_j = 0.0
            count_correct_j = 0
            count_incorrect_j = 0

            # Collect sums of probabilities for label_j
            for pred_dict, true_label in zip(predictions, targets):
                p_j = pred_dict.get(label_j, 0.0)
                if true_label == label_j:
                    sum_correct_j += p_j
                    count_correct_j += 1
                else:
                    sum_incorrect_j += p_j
                    count_incorrect_j += 1

            # Compute mean(prob) among correct vs. incorrect
            # Add eps to avoid dividing by zero.
            mean_correct = (sum_correct_j + eps) / (count_correct_j + eps)
            mean_incorrect = (sum_incorrect_j + eps) / (count_incorrect_j + eps)

            # Ratio = (mean_incorrect / mean_correct)
            ratio = mean_incorrect / mean_correct

            # Transform ratio with log-sigmoid:
            # log_ratio = ln(ratio), then apply the logistic function:
            #        1 / (1 + exp(-alpha * log_ratio))
            # which is mathematically = ratio^alpha / (1 + ratio^alpha).
            # We add an eps inside log() as well.
            log_ratio = math.log(ratio + eps)
            ratio_sigmoid = 1.0 / (1.0 + math.exp(-alpha * log_ratio))  # in (0,1)

            R_values.append(ratio_sigmoid)

        if not R_values:
            return None

        # Average the label-wise results
        return sum(R_values) / len(R_values)
