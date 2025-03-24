from ppm_benchmark.models.base_task_generator import BaseTaskGenerator


class OutcomeClassification(BaseTaskGenerator):

    def __init__(self):
        super().__init__()

    def _find_baseline_predictions(self, train):
        # TODO: More advanced baselines, maybe with sequences.
        return train['target'].value_counts(normalize=True).to_dict()

    def generate_task(self, train, test, task_name):
        categorical_case_attributes, categorical_event_attributes, numerical_case_attributes, numerical_event_attributes = self._classify_attributes(train)

        train_attr_values = self._find_train_attr_values(train, categorical_case_attributes + categorical_event_attributes)

        test_sequence_branches = self._find_closest_train_act_sequence(train, test)

        baseline_prediction = self._find_baseline_predictions(train)

        def process_group(group):
            group = group.reset_index(drop=True)
            activity_sequences = [tuple(group['concept:name'][:i + 1]) for i in range(len(group))]

            eval_cases = []
            for i, sequence in enumerate(activity_sequences):
                evaluation_dict = {
                    'task_name': task_name,
                    'case_id': group['case:concept:name'].iloc[0],
                    'test_index': group.index[i],
                    'train_branches': test_sequence_branches[sequence][0],
                    'closest_train_sequence': test_sequence_branches[sequence][1],
                    'train_sequence_distance': test_sequence_branches[sequence][2],
                    'prediction_target': group['target'].iloc[i],
                    'naive_baseline': baseline_prediction,
                    'fraction_completed': group['fraction_complete'].iloc[i],
                    'outcome_satisfied': group['outcome_satisfied'].iloc[i]
                }

                for attr in categorical_event_attributes + categorical_case_attributes:
                    row_attr_value = group[attr].iloc[i]
                    if row_attr_value not in train_attr_values[attr]:
                        evaluation_dict[f'attr_drift_{attr}'] = row_attr_value

                eval_cases.append(evaluation_dict)

            return eval_cases

        grouped = test.groupby('case:concept:name')
        eval_cases = []
        for _, group in grouped:
            eval_cases.extend(process_group(group))

        return eval_cases, {'naive_baseline': 'naive_baseline'}, True

