import pandas as pd
from .plot_generator import PlotGenerator
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class Evaluator:

    def __init__(self, evaluation_dicts, metrics, baselines, use_probas):
        self.evaluation_df = pd.DataFrame(evaluation_dicts)
        self.metrics = metrics
        self.use_probas = use_probas
        self.baselines = baselines
        self.plot_generator = PlotGenerator()

        self.init_baseline_preds(baselines)
        self.color_mappings = dict()
        self.models = []

    def init_baseline_preds(self, baselines):
        for baseline_name in baselines.keys():
            baseline_col_name = baselines[baseline_name]

            bl_series = self.evaluation_df[baseline_col_name].apply(pd.Series)
            bl_series = bl_series.add_prefix(f'{baseline_name}_')
            self.evaluation_df[f'{baseline_name}'] = self.evaluation_df[baseline_col_name]
            self.evaluation_df = pd.concat([self.evaluation_df, bl_series], axis=1)
        return

    def add_predictions(self, task_name, predictions, model_name, drop_type, drop_indices=None):
        if model_name not in self.models:
            self.models.append(model_name)

        pred_rows = []
        if self.use_probas:
            for prediction_probas in predictions:
                pred_row = dict()
                for target_name, proba in prediction_probas.items():
                    pred_row[f'{model_name}_{target_name}'] = proba

                pred_row[f'{model_name}'] = prediction_probas
                pred_rows.append(pred_row)
        else:
            pred_rows = [{f'{model_name}': pred} for pred in predictions]

        task_df = self.evaluation_df[self.evaluation_df['task_name'] == task_name].copy()
        print(f'Length before: {len(task_df)}')
        if drop_type == 'first':
            pred_mask = task_df.groupby('case_id')['test_index'].transform('min') != task_df['test_index']
        elif drop_type == 'last':
            pred_mask = task_df.groupby('case_id')['test_index'].transform('max') != task_df['test_index']
        elif drop_type == 'from_indices':
            pred_mask = ~task_df.reset_index().index.isin(drop_indices)
        else:
            pred_mask = np.ones(len(task_df), dtype=bool)

        pred_indices = task_df.loc[pred_mask].index
        print(f'Length after: {len(pred_indices)}')
        preds_df = pd.DataFrame(pred_rows, index=pred_indices)
        for column in preds_df.columns:
            if column in self.evaluation_df.columns:
                self.evaluation_df.loc[preds_df.index, column] = preds_df[column]
            else:
                self.evaluation_df[column] = np.nan
                self.evaluation_df.loc[preds_df.index, column] = preds_df[column]

        self.update_colors()
        return

    def evaluate(self):
        plot_items = self.models + [name for name in self.baselines.values()]
        plot_items = [f'{item}' for item in plot_items]
        self.evaluation_df = self.evaluation_df.dropna(subset=plot_items)
        tasks = self.evaluation_df['task_name'].unique()
        results = []

        for task in tasks:
            eval_df = self.evaluation_df[self.evaluation_df['task_name'] == task].reset_index(drop=True)
            for metric in self.metrics:
                performance_values = dict()
                for model in self.models:
                    model_metric_value = metric.evaluate(eval_df[f'{model}'], eval_df['prediction_target'])
                    performance_values[model] = model_metric_value
                for baseline_col in self.baselines.values():
                    baseline_metric_value = metric.evaluate(eval_df[f'{baseline_col}'], eval_df['prediction_target'])
                    performance_values[baseline_col] = baseline_metric_value

                result_dict = {
                    'task_name': task,
                    'metric': metric.name,
                }
                result_dict.update(performance_values)
                results.append(result_dict)
        return pd.DataFrame(results)

    def update_colors(self):
        baseline_colors = ['#AFCBFF', '#87CFFF', '#70B7FF']
        model_colors = ['#FF7043', '#00B36F', '#006DFF']

        for i, baseline in enumerate(self.baselines.values()):
            self.color_mappings[baseline] = baseline_colors[i]

        for i, model in enumerate(self.models):
            self.color_mappings[model] = model_colors[i]

    def get_metric(self, metric_name):
        plot_metric = None
        for metric in self.metrics:
            if metric.name == metric_name:
                plot_metric = metric
                break
        return plot_metric

    def init_plot(self, metric_name, models, save, use_pgf, single_row, task_type='XXXXXX'):
        self.plot_generator.initialize(task_type, save, use_pgf, single_row)
        plot_metric = self.get_metric(metric_name)
        print(f'plot_df length before dropping nan: {len(self.evaluation_df)}')
        plot_df = self.evaluation_df.dropna(subset=models).copy()
        print(f'plot_df length after dropping nan: {len(plot_df)}')
        #if 'outcome_satisfied' in plot_df.columns:
        #    plot_df.drop(plot_df[plot_df['outcome_satisfied'] == True].index, inplace=True).copy()
        inv_baselines = {v: k for k, v in self.baselines.items()}
        plot_names = [inv_baselines[item] if item in inv_baselines else item for item in models]

        plot_dict = dict()
        for model_name, plot_name in zip(models, plot_names):
            plot_dict[f'{model_name}'] = f'{plot_name}'

        return plot_df, plot_metric, plot_dict

    def _get_file_extension(self, use_pgf):
        if (use_pgf):
            file_extension = 'pgf'
        else:
            file_extension = 'png'
        return file_extension

    def plot_attr_drift(self, metric_name, task_type, save=True, use_pgf=False, single_row=False):
        plot_file_name = f'attr_drift_{task_type}_{metric_name}_{"_".join(self.models)}.{self._get_file_extension(use_pgf)}'
        plot_items = self.models + [name for name in self.baselines.values()]
        plot_df, plot_metric, plot_dict = self.init_plot(metric_name, plot_items, save, use_pgf, single_row, task_type)
        self.plot_generator.plot_attr_drift(plot_metric, self.evaluation_df, self.color_mappings, plot_dict, plot_file_name)

    def plot_by_train_distance(self, metric_name, task_type, save=True, use_pgf=False, single_row=False):
        plot_file_name = f'train_distance_{task_type}_{metric_name}_{"_".join(self.models)}.{self._get_file_extension(use_pgf)}'
        plot_items = self.models + [name for name in self.baselines.values()]
        plot_df, plot_metric, plot_dict = self.init_plot(metric_name, plot_items, save, use_pgf, single_row, task_type)
        self.plot_generator.plot_by_train_distance(plot_metric, plot_df, plot_dict, self.color_mappings, plot_file_name)

    def plot_by_fraction_completed(self, metric_name, task_type, save=True, use_pgf=False, single_row=False):
        plot_file_name = f'fraction_completed_{task_type}_{metric_name}_{"_".join(self.models)}.{self._get_file_extension(use_pgf)}'
        plot_items = self.models + [name for name in self.baselines.values()]
        plot_df, plot_metric, plot_dict = self.init_plot(metric_name, plot_items, save, use_pgf, single_row, task_type)
        self.plot_generator.plot_by_fraction_completed(plot_metric, plot_df, plot_dict, self.color_mappings, plot_file_name)
