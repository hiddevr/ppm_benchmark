from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import matplotlib
import math
from matplotlib.ticker import FormatStrFormatter
import os
from pathlib import Path


class PlotGenerator:

    def __init__(self):
        self.cmunbx_path = None
        self.cmunobx_path = None
        self.use_pgf = False

    def initialize(self, task_type, save, use_pgf, single_row):
        package_root = Path(__file__).resolve().parent.parent
        fonts_dir = package_root / "utils" / "computer_modern"
        self.cmunbx_path = os.path.join(fonts_dir, "cmunbx.ttf")
        self.cmunobx_path = os.path.join(fonts_dir, "cmunobx.ttf")
        self.use_pgf = use_pgf

        if save and use_pgf:
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })
        elif save:
            matplotlib.use("Agg")
            matplotlib.rcParams.update({
                'font.family': 'serif',
            })

        self.tick_color = '#2B2B2B'
        self.subtitle_color = '#7D7D7D'
        self.main_title_color = '#2B2B2B'
        self.axis_color = '#2B2B2B'
        self.single_row = single_row
        self.save = save

        self.title_prop = fm.FontProperties(
            fname=self.cmunbx_path,
            size=16 if not single_row else 18
        )
        self.plot_title_prop = fm.FontProperties(
            fname=self.cmunbx_path,
            size=10 if not single_row else 14
        )
        self.axis_prop = fm.FontProperties(
            fname=self.cmunbx_path,
            size=14 if not single_row else 16
        )
        self.legend_prop = fm.FontProperties(
            fname=self.cmunbx_path,
            size=12 if not single_row else 14
        )

        self.task_type = task_type

    def set_tick_font(self, ax, label_size=8):
        tick_prop = fm.FontProperties(
            fname=self.cmunobx_path,
            size=label_size
        )
        ax.tick_params(axis='both', labelsize=label_size, colors=self.tick_color)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(tick_prop)
            label.set_color(self.tick_color)
        
        for spine in ax.spines.values():
            spine.set_edgecolor(self.tick_color)

        return ax

    def set_axis_titles(self, fig, y1_title, x_title, y2_title=None, n_rows=1):
        """
        Set shared axis titles, with a small tweak if we only have 1 row
        (so that the x-label won't overlap ticks).
        """
        if self.use_pgf:
            y1_title = y1_title.replace('_', r'\_')
            x_title = x_title.replace('_', r'\_')

        # If only one row of subplots, push the shared x label a bit lower.
        if n_rows == 1:
            supx_y = 0
        else:
            supx_y = 0.025

        fig.supxlabel(
            x_title, 
            color=self.axis_color,
            fontproperties=self.axis_prop,
            y=supx_y
        )
        
        fig.supylabel(
            y1_title, 
            color=self.axis_color,
            fontproperties=self.axis_prop,
            x=0
        )

        if y2_title:
            if self.use_pgf:
                y2_title = y2_title.replace('_', r'\_')
            fig.text(
                0.96,
                0.5,
                y2_title,
                va='center',
                rotation='vertical',
                color=self.axis_color,
                fontproperties=self.axis_prop,
            )
        return fig
    
    def set_subplot_title(self, ax, task_name):
        if self.use_pgf:
            updated_task_name = task_name.replace(f'_{self.task_type}', '').replace('_', r'\_')
        else:
            updated_task_name = task_name.replace(f'_{self.task_type}', '')
        ax.set_title(
            updated_task_name,
            fontproperties=self.plot_title_prop,
            color=self.subtitle_color
        )
        return ax
    
    def set_legend(self, fig, handles, labels, ncol):
        """
        Put the legend at the top center so it doesn't overlap subplots.
        """
        y_anchor = 1.02 if self.single_row else 0.95
        legend = fig.legend(
            handles,
            [label.replace('_', r'\_') for label in labels if self.use_pgf],
            loc='upper center',
            ncol=ncol,
            bbox_to_anchor=(0.5, y_anchor),
            frameon=False,
            labelcolor=self.axis_color,
            prop=self.legend_prop
        )
        return fig
    
    def set_main_title(self, fig, title):
        fig.suptitle(
            title, 
            fontproperties=self.title_prop,
            color=self.main_title_color,
        )
        return fig

    def plot_attr_drift(self, metric, plot_df, color_mappings, plot_dict, file_name):
        drift_columns = [col for col in plot_df.columns if col.startswith('attr_drift_')]
        unique_tasks = plot_df['task_name'].unique()

        # Gather penalties
        penalties = {
            task: {plot_col: {} for plot_col in plot_dict.keys()}
            for task in unique_tasks
        }

        model_scores = defaultdict(dict)
        for task in unique_tasks:
            for model in plot_dict.keys():
                try:
                    task_df = plot_df.loc[plot_df['task_name'] == task].dropna(subset=[model])
                    model_scores[model][task] = metric.evaluate(task_df[model], task_df['prediction_target'])
                except Exception as e:
                    print(f'An error occured calculating predictions for {model} {task}: {e}')

        for task in unique_tasks:
            task_df = plot_df.loc[plot_df['task_name'] == task]
            for col in drift_columns:
                attribute_name = col.replace('attr_drift_', '')
                for model in plot_dict.keys():
                    drift_df = task_df.dropna(subset=[col, model])
                    if len(drift_df) > 0:
                        try:
                            performance = metric.evaluate(
                                drift_df[model],
                                drift_df['prediction_target']
                            )
                            penalty = model_scores[model][task] - performance
                        except Exception as e:
                            print(e)
                            penalty = np.nan
                    else:
                        penalty = np.nan 
                    penalties[task][model][attribute_name] = penalty

        # Filter out tasks that don't have complete data
        filtered_tasks = []
        for task in unique_tasks:
            plot_cols_data = penalties[task]
            has_all_plot_cols_data = all(
                any(not np.isnan(perf) for perf in plot_cols_data[plot_col].values())
                for plot_col in plot_dict.keys()
            )
            if has_all_plot_cols_data:
                filtered_tasks.append(task)
        
        unique_tasks = filtered_tasks
        if len(unique_tasks) == 0:
            print("No tasks have complete data across all plot_cols. Nothing to plot.")
            return

        # Figure layout
        n_cols = 3
        n_rows = math.ceil(len(unique_tasks) / n_cols)
        
        subplot_width = 4
        subplot_height = 3

        # If we have only one row of plots, let's increase the figure size a bit
        if n_rows == 1:
            subplot_width = 5
            subplot_height = 4

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(subplot_width * n_cols, subplot_height * n_rows),
            sharex=False
        )

        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        first_ax = None

        bar_height = 0.5
        group_spacing = 0.8

        for i, task in enumerate(unique_tasks):
            ax = axes[i]

            all_attributes = list(penalties[task][list(plot_dict.keys())[0]].keys())
            # Filter out attributes that have no data across any plot_col
            attributes = []
            for attr in all_attributes:
                if any(
                    not np.isnan(penalties[task][plot_col][attr])
                    for plot_col in plot_dict.keys()
                ):
                    attributes.append(attr)

            if not attributes:
                ax.set_visible(False)
                continue
            
            num_attributes = len(attributes)
            group_height = bar_height * len(plot_dict)
            group_positions = np.arange(num_attributes) * (group_height + group_spacing)

            for j, plot_col in enumerate(plot_dict.keys()):
                penalty_values = [
                    penalties[task][plot_col][attr] 
                    if not np.isnan(penalties[task][plot_col][attr]) else 0
                    for attr in attributes
                ]
                bar_positions = group_positions + j * bar_height

                ax.barh(
                    bar_positions,
                    penalty_values,
                    height=bar_height,
                    align='center',
                    label=plot_dict[plot_col],
                    color=color_mappings[plot_col]
                )

            ax.set_yticks(group_positions + (group_height - bar_height) / 2)
            ax.set_yticklabels([attr.replace('_', '\_') for attr in attributes])
            ax.invert_yaxis()

            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax = self.set_subplot_title(ax, task)
            ax = self.set_tick_font(ax, label_size=8 if n_rows > 1 else 10)
            if i == 0:
                first_ax = ax

        # Hide any empty subplots
        for j in range(len(unique_tasks), n_rows * n_cols):
            fig.delaxes(axes[j])

        if first_ax is not None:
            handles, labels = first_ax.get_legend_handles_labels()
        else:
            handles, labels = [], []

        # Shared axis labels and legend
        fig = self.set_axis_titles(
            fig,
            y1_title='Drift Attribute',
            x_title=f"{metric.name} Penalty",
            n_rows=n_rows
        )
        fig = self.set_legend(fig, handles, labels, ncol=len(plot_dict))

        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.90], pad=2.5)
        fig.subplots_adjust(top=0.85, bottom=0.15, right=0.90)

        if self.save:
            plt.savefig(f'plots/{file_name}',
                        transparent=True, dpi=600)
        plt.show()

    def plot_by_train_distance(self, metric, plot_df, plot_dict, color_mappings, plot_file_name):
        unique_tasks = plot_df['task_name'].unique()
        n_cols = 3
        n_rows = math.ceil(len(unique_tasks) / n_cols)
        
        subplot_width = 4
        subplot_height = 3
        if n_rows == 1:
            subplot_width = 5
            subplot_height = 4

        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(subplot_width * n_cols, subplot_height * n_rows),
            sharex=False,
            sharey=False
        )
        
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        first_ax2 = None
        
        for i, task in enumerate(unique_tasks):
            print(f'Generating plot for {task}')
            ax = axes[i]
            task_df = plot_df[plot_df['task_name'] == task]

            train_distances = sorted([d for d in task_df['train_sequence_distance'].unique()])
            eval_results = {eval_col: [] for eval_col in plot_dict.keys()}
            sample_fractions = []
        
            total_samples = len(task_df)
            print(f'Total original traces in test data: {total_samples}')
            samples_accounted_for = 0
            for dist in train_distances:
                sub_df = task_df[task_df['train_sequence_distance'] == dist]
                sample_fractions.append(len(sub_df) / total_samples)
                samples_accounted_for += len(sub_df)
                for eval_col in plot_dict.keys():
                    eval_results[eval_col].append(
                        metric.evaluate(sub_df[eval_col], sub_df['prediction_target'])
                    )
            print(f'Number of samples included: {samples_accounted_for}')
            
            best_performance = 0
            least_performance = float('inf')
            for eval_col, performances in eval_results.items():
                ax.plot(
                    train_distances,
                    performances,
                    marker='o',
                    label=plot_dict[eval_col],
                    color=color_mappings[eval_col],
                    markersize=2
                )
                for performance in performances:
                    best_performance = max(best_performance, performance)
                    least_performance = min(least_performance, performance)

            # Buffer on y-limits
            if least_performance < float('inf'):
                ax.set_ylim([least_performance * 1.01, best_performance * 1.01])
            
            # Plot fraction of samples on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(
                train_distances,
                sample_fractions,
                color=self.subtitle_color,
                linestyle='--',
                marker='o',
                markersize=2,
                label='Fraction of Total Samples'
            )
            ax2.tick_params(axis='y', colors=self.tick_color)

            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax = self.set_subplot_title(ax, task)
            ax2 = self.set_tick_font(ax2, label_size=6 if n_rows > 1 else 8)
            ax = self.set_tick_font(ax, label_size=6 if n_rows > 1 else 8)

            if i == 0:
                first_ax2 = ax2

        for j in range(len(unique_tasks), n_rows * n_cols):
            fig.delaxes(axes[j])
        
        # Collect handles for legend from first axes
        handles, labels = axes[0].get_legend_handles_labels()
        if first_ax2:
            sample_handle, sample_label = first_ax2.get_legend_handles_labels()
            handles += sample_handle
            labels += sample_label
        
        fig = self.set_axis_titles(
            fig,
            y1_title=metric.name,
            x_title='Train Sequence Distance',
            y2_title='Fraction of Total Samples',
            n_rows=n_rows
        )
        fig = self.set_legend(fig, handles, labels, ncol=len(plot_dict) + 1)
        
        fig.tight_layout(rect=[0, 0, 1, 0.90], pad=2.5)
        fig.subplots_adjust(top=0.85, bottom=0.15, right=0.90)

        if self.save:
            plt.savefig(f'plots/{plot_file_name}',
                        transparent=True, dpi=600)
        plt.show()

    def plot_by_fraction_completed(self, metric, plot_df, plot_dict, color_mappings, file_name):
        unique_tasks = plot_df['task_name'].unique()
        n_cols = 3
        n_rows = math.ceil(len(unique_tasks) / n_cols)
        
        subplot_width = 4
        subplot_height = 3
        if n_rows == 1:
            subplot_width = 5
            subplot_height = 4

        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(subplot_width * n_cols, subplot_height * n_rows),
            sharex=False
        )
        
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        first_ax = None
        
        bin_edges = np.arange(0, 1.05, 0.05)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_rounded = np.round(bin_centers, 3)
        
        for i, task in enumerate(unique_tasks):
            ax = axes[i]
            task_df = plot_df[plot_df['task_name'] == task].copy()
            
            # Bin fraction_completed in increments of 0.05
            task_df['bin'] = pd.cut(
                task_df['fraction_completed'], 
                bins=bin_edges, 
                labels=bin_centers, 
                include_lowest=True
            )
            
            plot_values = {eval_col: [] for eval_col in plot_dict.keys()}
            grouped = task_df.groupby('bin')

            for bin_val in bin_centers:
                try:
                    group = grouped.get_group(bin_val)
                except KeyError:
                    group = pd.DataFrame()

                for eval_col in plot_dict.keys():
                    if not group.empty:
                        try:
                            eval_result = metric.evaluate(group[eval_col], group['prediction_target'])
                        except Exception as e:
                            print(f"Error evaluating metric for task '{task}', bin '{bin_val}': {e}")
                            eval_result = np.nan
                    else:
                        eval_result = np.nan
                    plot_values[eval_col].append(eval_result)
            
            x = np.arange(len(bin_centers))
            width = 0.8 / len(plot_dict)

            # Plot each column side-by-side in each bin
            for j, (eval_col, values) in enumerate(plot_values.items()):
                ax.bar(
                    x + j * width,
                    values,
                    width,
                    label=plot_dict[eval_col],
                    color=color_mappings[eval_col]
                )
            
            valid_values = [val for vals in plot_values.values() for val in vals if not np.isnan(val)]
            if valid_values:
                least_performance = min(valid_values) * 0.99
                best_performance = max(valid_values) * 1.01
                ax.set_ylim([least_performance, best_performance])
            else:
                ax.set_ylim([0, 1])
            
            # X‐ticks at bin centers
            ax.set_xticks(x + width * (len(plot_dict) - 1) / 2)
            ax.set_xticklabels([f'{bc:.2f}' for bc in bin_centers_rounded], rotation=45, ha='center')

            ax = self.set_subplot_title(ax, task)
            ax = self.set_tick_font(ax, label_size=6 if n_rows > 1 else 10)
            
            if i == 0:
                first_ax = ax
        
        for j in range(len(unique_tasks), n_rows * n_cols):
            fig.delaxes(axes[j])
        
        # Legend
        if first_ax is not None:
            handles, labels = first_ax.get_legend_handles_labels()
        else:
            handles, labels = [], []

        fig = self.set_axis_titles(
            fig,
            y1_title=metric.name,
            x_title='Fraction Completed',
            n_rows=n_rows
        )
        fig = self.set_legend(fig, handles, labels, ncol=len(plot_dict))
        
        fig.tight_layout(rect=[0, 0, 1, 0.85], pad=2.5)
        fig.subplots_adjust(top=0.85, bottom=0.15, right=0.90)
        
        if self.save:
            plt.savefig(f'plots/{file_name}',
                        transparent=True, dpi=600)
        plt.show()
