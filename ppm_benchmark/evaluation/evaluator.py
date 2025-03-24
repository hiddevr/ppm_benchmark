import pandas as pd
from ppm_benchmark.evaluation.plot_generator import PlotGenerator


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

    def add_predictions(self, task_name, predictions, model_name):
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

        task_indices = self.evaluation_df[self.evaluation_df['task_name'] == task_name].index
        preds_df = pd.DataFrame(pred_rows, index=task_indices)
        for column in preds_df.columns:
            if column in self.evaluation_df.columns:
                self.evaluation_df.loc[preds_df.index, column] = preds_df[column]
            else:
                self.evaluation_df[column] = preds_df[column]

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
        colors = ['blue', 'red', 'green', 'orange', 'yellow']
        baselines = [name for name in self.baselines.values()]
        legend_items = self.models + baselines
        self.color_mappings = {model: color for model, color in zip(legend_items, colors)}

    def get_metric(self, metric_name):
        plot_metric = None
        for metric in self.metrics:
            if metric.name == metric_name:
                plot_metric = metric
                break
        return plot_metric

    def init_plot(self, metric_name, plot_items):
        plot_metric = self.get_metric(metric_name)
        plot_df = self.evaluation_df.dropna(subset=plot_items).copy()
        #if 'outcome_satisfied' in plot_df.columns:
        #    plot_df.drop(plot_df[plot_df['outcome_satisfied'] == True].index, inplace=True).copy()
        inv_baselines = {v: k for k, v in self.baselines.items()}
        plot_names = [inv_baselines[item] if item in inv_baselines else item for item in plot_items]

        plot_dict = dict()
        for item_name, plot_name in zip(plot_items, plot_names):
            plot_dict[f'{item_name}'] = f'{plot_name}'

        return plot_df, plot_metric, plot_dict

    def plot_attr_drift(self, metric_name):
        plot_items = self.models + [name for name in self.baselines.values()]
        plot_df, plot_metric, plot_dict = self.init_plot(metric_name, plot_items)
        self.plot_generator.plot_attr_drift(plot_metric, self.evaluation_df, self.color_mappings, plot_dict)

    def plot_by_train_distance(self, metric_name):
        plot_items = self.models + [name for name in self.baselines.values()]
        plot_df, plot_metric, plot_dict = self.init_plot(metric_name, plot_items)
        print(plot_df)
        self.plot_generator.plot_by_train_distance(plot_metric, plot_df, plot_dict, self.color_mappings)

    def plot_by_fraction_completed(self, metric_name):
        plot_items = self.models + [name for name in self.baselines.values()]
        plot_df, plot_metric, plot_dict = self.init_plot(metric_name, plot_items)
        print(plot_df)
        self.plot_generator.plot_by_fraction_completed(plot_metric, plot_df, plot_dict, self.color_mappings)
