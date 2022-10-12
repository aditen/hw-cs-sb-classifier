import os.path

import pandas as pd
from tabulate import tabulate

from data_augmentation import DataAugmentationOptions
from data_loading import DataloaderKinderlabor
from grayscale_model import ModelVersion
from training import TrainerKinderlabor, LossFunction
from utils import data_split_dict, short_names_models, short_names_tasks, TaskType, DataSplit
from visualizing import VisualizerKinderlabor

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
    if os.path.isfile(csv_path):
        VisualizerKinderlabor.visualize_baseline_results_as_plot(csv_path)
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
