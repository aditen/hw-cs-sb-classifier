import os.path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from tabulate import tabulate
import seaborn as sns

from data_augmentation import DataAugmentationOptions
from data_loading import DataloaderKinderlabor
from grayscale_model import ModelVersion
from training import TrainerKinderlabor, LossFunction
from visualizing import VisualizerKinderlabor
from utils import data_split_dict, short_names_models, short_names_tasks, TaskType, DataSplit, long_names_tasks, \
    long_names_models

run_configs = [
    ("none", DataAugmentationOptions.none_aug()),
    ("ac", DataAugmentationOptions.auto_ctr_aug()),
    ("eq", DataAugmentationOptions.eq_aug()),
    ("geo", DataAugmentationOptions.geo_aug()),
    ("geo_ac", DataAugmentationOptions.geo_ac_aug()),
    ("crop", DataAugmentationOptions.crop_aug()),
    ("crop_plus", DataAugmentationOptions.crop_plus_aug()),
]


def get_run_id(task_type: TaskType, aug_name: str, data_split: DataSplit, model: ModelVersion):
    return f"base_task[{short_names_tasks[task_type]}]_aug[{aug_name}]" \
           f"_split[{data_split_dict[data_split]}]_model[{short_names_models[model]}]"


csv_path = './output_visualizations/runs_base.csv'
if __name__ == "__main__":
    # TODO: move to run utils or visualizing
    if os.path.isfile(csv_path):
        # see https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        pd.options.mode.chained_assignment = None
        full_df = pd.read_csv(csv_path, sep=";")
        sns.set_theme(font_scale=1.75, style='whitegrid')
        fig = plt.figure(figsize=(21., 19.2))
        all_tasks = list(TaskType)
        all_models = [ModelVersion.SM, ModelVersion.LE_NET]
        grid = plt.GridSpec(len(all_tasks), 1)

        for task_type in all_tasks:
            fake = fig.add_subplot(grid[all_tasks.index(task_type)])
            # '\n' remains important
            fake.set_title(f'{long_names_tasks[task_type]}\n', fontweight='semibold', size=28)
            fake.set_axis_off()

            gs = GridSpecFromSubplotSpec(1, len(all_models),
                                         subplot_spec=grid[all_tasks.index(task_type)])

            model_idx = 0
            for i, model in enumerate(all_models):
                ax = fig.add_subplot(gs[model_idx])
                df_loop = full_df[(full_df['model'] == model.name) & (full_df['task'] == short_names_tasks[task_type])]
                df_loop.performance *= 100
                sns.barplot(
                    data=df_loop, errorbar=None,
                    x="split", y="performance", hue="augmentation",
                    alpha=.6, ax=ax
                )
                # not needed because there is enough visual space
                for j, (tick) in enumerate(ax.xaxis.get_major_ticks()):
                    if (j % 2) != (i % 2):
                        pass
                        # tick.set_visible(False)
                ax.set_ylim(0, 100)
                ax.set_title(f'{long_names_models[model]}', size=25)
                ax.get_legend().remove()

                ax.set_xlabel('Data Split')
                if model_idx == 0:
                    ax.set_ylabel('Balanced Accuracy (%)')
                else:
                    ax.set_ylabel(None)
                model_idx += 1

        # add legend
        handles, labels = fig.axes[2].get_legend_handles_labels()
        fig.axes[2].legend(handles, labels, bbox_to_anchor=(1, 1.03), title="Augmentation")
        fig.tight_layout()
        plt.show()
        fig.savefig("./output_visualizations/base_plot.pdf", dpi=75)
    else:
        data_arr = []
        for task_type in TaskType:
            for run_config in run_configs:
                for data_split in DataSplit:
                    for model in [ModelVersion.SM, ModelVersion.LE_NET]:
                        run_id = get_run_id(task_type, run_config[0], data_split, model)
                        print(
                            f'Running for augmentation {run_config[0]} and data split {data_split.value} '
                            f'and task {task_type.name} using model {model.name}')

                        # Initialize data loader: data splits and loading from images from disk
                        loader = DataloaderKinderlabor(task_type=task_type,
                                                       data_split=data_split,
                                                       augmentation_options=run_config[1])

                        # visualize class distribution and some (train) samples
                        visualizer = VisualizerKinderlabor(loader, run_id=run_id)

                        # Train model and analyze training progress (mainly when it starts overfitting on validation set)
                        trainer = TrainerKinderlabor(loader, load_model_from_disk=True,
                                                     run_id=run_id,
                                                     loss_function=LossFunction.SOFTMAX if task_type != TaskType.CROSS else LossFunction.BCE,
                                                     model_version=model)
                        trainer.train_model(n_epochs=75)
                        visualizer.visualize_training_progress(trainer)

                        # Predict on test samples
                        trainer.predict_on_test_samples()
                        visualizer.visualize_model_errors(trainer)
                        trainer.script_model()

                        data_arr.append(
                            [run_config[0], data_split_dict[data_split], short_names_tasks[task_type],
                             trainer.get_predictions()[6], model.name])
        print(tabulate([[x[1], x[2], x[4], x[0], f'{(x[3] * 100):.2f}%'] for x in data_arr],
                       headers=['Data Split', 'Task', 'Model', 'Augmentation', 'Performance']))
        df = pd.DataFrame.from_dict({'augmentation': [row[0] for row in data_arr],
                                     'split': [row[1] for row in data_arr],
                                     'task': [row[2] for row in data_arr],
                                     'performance': [row[3] for row in data_arr],
                                     'model': [row[4] for row in data_arr]})
        df.to_csv('./output_visualizations/runs_base.csv', sep=';')
