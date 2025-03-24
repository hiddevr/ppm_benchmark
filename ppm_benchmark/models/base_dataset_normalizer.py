from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from ..utils.outcome_ltl_parser import EvaluatorWithPosition, Lexer, Parser
from ..utils.el_splitter import ELSplitter
from ..utils.logger import setup_logger
import random

logger = setup_logger(__name__)


class BaseDatasetNormalizer(ABC):

    def __init__(self):
        pass

    def calc_next_attribute_target(self, df, attr_col):
        def calcNextEvent(grp):
            grp["target"] = grp[attr_col].shift(periods=-1)
            return grp

        df = df.groupby("case:concept:name").apply(calcNextEvent)

        if attr_col == 'time:timestamp':
            df['target'] = df['target'].astype('int64').astype(int) / (24 * 60 * 60)
        df = df.reset_index(drop=True)
        return df

    def calc_outcome_target(self, df, ltl_rule):

        if ltl_rule == 'REMAINING_TIME':
            df["target"] = df.groupby("case:concept:name")["time:timestamp"].apply(
                lambda x: x.max() - x).values
            df["target"] = df["target"].dt.total_seconds() / (24 * 60 * 60)

        else:
            grouped_cases = df.groupby('case:concept:name')

            evaluator = EvaluatorWithPosition(grouped_cases)
            lexer = Lexer(ltl_rule)
            parser = Parser(lexer.tokens)
            ast = parser.parse()

            results = {}
            df['outcome_satisfied'] = False

            for case, group in grouped_cases:
                activities = group['concept:name'].tolist()
                satisfied, position = evaluator.evaluate_with_position(ast, activities)
                results[case] = (satisfied, position)
                if satisfied:
                    df.loc[group.index[position:], 'outcome_satisfied'] = True

            df['target'] = df.groupby('case:concept:name')['outcome_satisfied'].transform(lambda x: x.any())

        return df

    def calc_attr_suffix_target(self, df, attr_col):
        def calc_suffix(group):
            activities = group[attr_col].tolist()
            targets = [tuple(activities[i + 1:]) for i in range(len(activities))]
            group['target'] = targets
            return group

        df = df.groupby("case:concept:name").apply(calc_suffix).reset_index(drop=True)
        return df

    @abstractmethod
    def normalize_next_attribute(self, df):
        pass

    @abstractmethod
    def normalize_outcome(self, df):
        pass

    @abstractmethod
    def normalize_attribute_suffix(self, df):
        pass

    def normalize_and_split(self, df, task_type, start_date, end_date, outcome_ltl_rule=None, attr_col=None):
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

        if task_type == 'next_attribute':
            df = self.normalize_next_attribute(df)
            df = self.calc_next_attribute_target(df, attr_col)
        elif task_type == 'outcome':
            df = self.normalize_outcome(df)
            df = self.calc_outcome_target(df, outcome_ltl_rule)
        elif task_type == 'attribute_suffix':
            df = self.normalize_attribute_suffix(df)
            df = self.calc_attr_suffix_target(df, attr_col)

        else:
            raise ValueError(f'Unknown task type {task_type}')

        logger.debug(f'Full df entries: {len(df)}')
        splitter = ELSplitter()
        train, test = splitter.debias_and_split(df, start_date, end_date)
        logger.debug(f'Train entries: {len(train)}')
        test = test[test['target'].notna()]
        logger.debug(f'Test entries: {len(test)}')

        return train, test
