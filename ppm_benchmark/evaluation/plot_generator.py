import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from ppm_benchmark.models.base_metric import BaseMetric


class PlotGenerator:

    def __init__(self):
        pass

    def plot_attr_drift(self, metric: BaseMetric, plot_df: pd.DataFrame, color_mappings: dict, plot_dict: dict):
        drift_columns = [col for col in plot_df.columns if col.startswith('attr_drift_')]
        unique_tasks = plot_df['task_name'].unique()

        penalties = {task: {} for task in unique_tasks}
        for task in penalties.keys():
            penalties[task] = {plot_col: {} for plot_col in plot_dict.keys()}

        for task in unique_tasks:
            task_df = plot_df.loc[plot_df['task_name'] == task]
            for col in drift_columns:
                drift_df = task_df.dropna(subset=[col])
                for plot_col in plot_dict.keys():
                    attribute_name = col.replace('attr_drift_', '')
                    if len(task_df) > 0:
                        performance = metric.evaluate(drift_df[plot_col],
                                                      drift_df['prediction_target'])
                    else:
                        performance = 0
                    penalties[task][plot_col][attribute_name] = performance

        fig, axes = plt.subplots(len(unique_tasks), 1, figsize=(10, 5 * len(unique_tasks)))

        if len(unique_tasks) == 1:
            axes = [axes]

        bar_height = 0.8  # Adjust this value to change the height of each bar
        group_height = bar_height * len(plot_dict)  # Total height for each group of bars

        for i, task in enumerate(unique_tasks):
            attributes = list(penalties[task][list(plot_dict.keys())[0]].keys())
            num_attributes = len(attributes)

            # Calculate the positions for the groups
            group_positions = np.arange(num_attributes) * (group_height + 0.5)

            for j, plot_col in enumerate(plot_dict.keys()):
                penalty_values = list(penalties[task][plot_col].values())

                # Calculate the position for each bar within the group
                bar_positions = group_positions + j * bar_height

                axes[i].barh(bar_positions, penalty_values, height=bar_height, align='center',
                             label=plot_dict[plot_col], color=color_mappings[plot_col])

            # Set the y-ticks to be in the middle of each group
            axes[i].set_yticks(group_positions + (group_height - bar_height) / 2)
            axes[i].set_yticklabels(attributes)

            axes[i].invert_yaxis()  # Labels read top-to-bottom
            axes[i].set_xlabel(f"{metric.name} Penalty")
            axes[i].set_title(task)

            # Adjust the y-axis limits to show all bars
            axes[i].set_ylim(group_positions[-1] + group_height, group_positions[0] - 0.5)

        fig.suptitle(f'{metric.name} Penalty by Drift Attribute for Each Task', fontsize=16)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.85), title="Legend")
        plt.tight_layout()
        plt.show()

    def plot_by_train_distance(self, metric, plot_df, plot_dict, color_mappings):
        unique_tasks = plot_df['task_name'].unique()
        fig, axes = plt.subplots(len(unique_tasks), 1, figsize=(10, 5 * len(unique_tasks)), sharex=False)

        if len(unique_tasks) == 1:
            axes = [axes]

        for i, task in enumerate(unique_tasks):
            task_df = plot_df[plot_df['task_name'] == task]
            train_distances = sorted([d for d in task_df['train_sequence_distance'].unique() if d <= 30])

            eval_results = {eval_col: [] for eval_col in plot_dict.keys()}
            sample_fractions = []

            total_samples = len(task_df)

            for dist in train_distances:
                sub_df = task_df[task_df['train_sequence_distance'] == dist]
                sample_fractions.append(len(sub_df) / total_samples)

                for eval_col in plot_dict.keys():
                    eval_results[eval_col].append(
                        metric.evaluate(sub_df[eval_col], sub_df['prediction_target']))

            for eval_col, performance in eval_results.items():
                axes[i].plot(train_distances, performance, marker='o', label=plot_dict[eval_col],
                             color=color_mappings[eval_col])

            axes[i].set_ylabel(f"{metric.name}")
            axes[i].set_title(task)
            axes[i].set_xlabel("Train Sequence Distance")
            axes[i].set_ylim([0, 1])

            # Add a secondary y-axis for the fraction of total samples
            ax2 = axes[i].twinx()
            ax2.plot(train_distances, sample_fractions, color='grey', linestyle='--', marker='o',
                     label='Fraction of Total Samples')
            ax2.set_ylabel('Fraction of Total Samples')

        fig.suptitle(f"Prediction {metric.name} by Train Sequence Distance", fontsize=16)

        # Adding legends for both axes
        lines, labels = axes[0].get_legend_handles_labels()
        sample_line, sample_label = ax2.get_legend_handles_labels()
        lines += sample_line
        labels += sample_label
        fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.85), title="Legend")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_lass_bar(self, metric, plot_df, plot_dict, color_mappings):
        unique_tasks = plot_df['task_name'].unique()

        fig, axes = plt.subplots(len(unique_tasks), 1, figsize=(1, 2 * len(unique_tasks)), sharex=False)

        if len(unique_tasks) == 1:
            axes = [axes]

        for i, task in enumerate(unique_tasks):
            task_df = plot_df[plot_df['task_name'] == task]

            eval_results = {eval_col: {} for eval_col in plot_dict.keys()}
            for eval_col in plot_dict.keys():
                eval_results[eval_col] = metric.evaluate(task_df[eval_col], task_df['prediction_target'])

            first_key = list(eval_results.keys())[0]
            x_labels = list(eval_results[first_key].keys())

            plot_values = {plot_item: list(eval_results[plot_item].values()) for plot_item in eval_results.keys()}

            x = np.arange(len(x_labels))

            for j, (plot_item, values) in enumerate(plot_values.items()):
                axes[i].scatter(x, values, label=plot_dict[plot_item], color=color_mappings[plot_item], s=5)

                # Add lines from x-axis to points
                for x_pos, val in zip(x, values):
                    axes[i].plot([x_pos, x_pos], [0, val], color=color_mappings[plot_item], alpha=0.3, linestyle='--',
                                 linewidth=0.5)

            axes[i].set_title(task, va='bottom')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(x_labels, rotation=90, ha='right')
            axes[i].tick_params(axis='x', labelsize=7)
            axes[i].set_ylim(bottom=0)  # Start y-axis at 0
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            axes[i].set_xlim(-0.5, len(x_labels) - 0.5)

        plot_width = len(axes[0].get_xticks()) / 3
        fig.set_size_inches(plot_width, 8)
        fig.suptitle("LASS for model predictions vs distance baseline", fontsize=16)
        fig.text(0.5, 0.02, 'Activities', ha='center', va='center')
        fig.text(0.02, 0.5, 'LASS Score', ha='center', va='center', rotation='vertical')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.85), title="Legend")

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.show()

    def plot_by_fraction_completed(self, metric, plot_df, plot_dict, color_mappings):
        unique_tasks = plot_df['task_name'].unique()

        fig, axes = plt.subplots(len(unique_tasks), 1, figsize=(10, 5 * len(unique_tasks)), sharex=True)

        if len(unique_tasks) == 1:
            axes = [axes]

        bin_edges = np.arange(0, 1.05, 0.05)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_rounded = np.round(bin_centers, 3)

        for i, task in enumerate(unique_tasks):
            task_df = plot_df[plot_df['task_name'] == task]

            task_df['bin'] = pd.cut(task_df['fraction_completed'], bins=bin_edges, labels=bin_centers)
            plot_values = {
                eval_col: task_df.groupby('bin').apply(lambda x: metric.evaluate(x[eval_col], x['prediction_target']))
                for eval_col in plot_dict.keys()}

            x = np.arange(len(bin_centers))
            width = 0.8 / len(plot_values)

            for j, (eval_col, values) in enumerate(plot_values.items()):
                axes[i].bar(x + j * width, values, width, label=plot_dict[eval_col], color=color_mappings[eval_col])

            axes[i].set_title(task)
            axes[i].set_xticks(x + width * (len(plot_values) - 1) / 2)
            axes[i].set_xticklabels([f'{bc:.3f}' for bc in bin_centers_rounded], rotation=45, ha='right')

        fig.suptitle(f"{metric.name} by Fraction Completed", fontsize=16)
        fig.text(0.5, 0.04, 'Fraction Completed', ha='center', va='center')
        fig.text(0.04, 0.5, f'{metric.name}', ha='center', va='center', rotation='vertical')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.85), title="Legend")

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        plt.show()