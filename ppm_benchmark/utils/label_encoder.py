import ast
from collections.abc import Sequence
import pandas as pd


class PPMLabelEncoder:
    def __init__(self):
        self.label_mappings = {}
        self.inverse_mappings = {}

    def _parse_if_tuple(self, value):
        # Use ast.literal_eval to convert string representations of tuples/lists back to actual tuples/lists
        try:
            parsed_value = ast.literal_eval(value)
            if isinstance(parsed_value, tuple):
                return parsed_value
        except (ValueError, SyntaxError):
            pass
        return value

    def fit(self, X):
        for column in X.columns:
            # Parse strings representing tuples
            parsed_col = X[column].apply(self._parse_if_tuple)
            unique_elements = set()

            # Vectorized collection of unique elements
            if isinstance(parsed_col.iloc[0], Sequence) and not isinstance(parsed_col.iloc[0], str):
                # Collect unique elements from tuples/lists
                unique_elements = {el for row in parsed_col for el in row}
            else:
                # Collect unique single values
                unique_elements = set(parsed_col.unique())

            # Create column-specific mappings
            label_mapping = {label: idx for idx, label in enumerate(unique_elements)}
            inverse_mapping = {idx: label for idx, label in enumerate(unique_elements)}

            # Store mappings specific to the current column
            self.label_mappings[column] = label_mapping
            self.inverse_mappings[column] = inverse_mapping

    def transform(self, X):
        transformed_X = X.copy()

        for column in X.columns:
            # Parse strings representing tuples
            parsed_col = transformed_X[column].apply(self._parse_if_tuple)

            if isinstance(parsed_col.iloc[0], Sequence) and not isinstance(parsed_col.iloc[0], str):
                # Vectorized transformation for tuple elements
                label_mapping = self.label_mappings[column]
                transformed_X[column] = parsed_col.apply(
                    lambda x: tuple(label_mapping[val] for val in x)
                )
            else:
                # Faster map for non-tuple values
                label_mapping = self.label_mappings[column]
                transformed_X[column] = parsed_col.map(label_mapping)

        return transformed_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform_with_new_labels(self, X):
        transformed_X = X.copy()

        for column in X.columns:
            parsed_col = transformed_X[column].apply(self._parse_if_tuple)
            label_mapping = self.label_mappings[column]
            inverse_mapping = self.inverse_mappings[column]

            if isinstance(parsed_col.iloc[0], Sequence) and not isinstance(parsed_col.iloc[0], str):
                # Add any new unseen labels for tuple elements
                for row in parsed_col:
                    for element in row:
                        if element not in label_mapping:
                            new_index = len(label_mapping)
                            label_mapping[element] = new_index
                            inverse_mapping[new_index] = element

                transformed_X[column] = parsed_col.apply(
                    lambda x: tuple(label_mapping[val] for val in x)
                )
            else:
                # Add new unseen labels for non-tuple values
                for label in parsed_col.unique():
                    if label not in label_mapping:
                        new_index = len(label_mapping)
                        label_mapping[label] = new_index
                        inverse_mapping[new_index] = label

                transformed_X[column] = parsed_col.map(label_mapping)

        return transformed_X

    def inverse_transform_column(self, column_name, values):
        if column_name not in self.inverse_mappings:
            raise ValueError(f"Column '{column_name}' not found in inverse mappings.")

        if isinstance(values[0], Sequence) and not isinstance(values[0], str):
            # Vectorized inverse transformation for tuple elements
            inverse_mapping = self.inverse_mappings[column_name]
            return [tuple(inverse_mapping[val] for val in value) for value in values]
        else:
            # Vectorized inverse transformation for single values
            inverse_mapping = self.inverse_mappings[column_name]
            return pd.Series(values).map(inverse_mapping).tolist()
