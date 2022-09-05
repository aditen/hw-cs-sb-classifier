import pandas as pd
from tabulate import tabulate

from data_augmentation import DataAugmentationOptions
from data_loading import DataloaderKinderlabor, TaskType, DataSplit
from grayscale_model import ModelVersion
from run_utils import RunUtilsKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor

run_configs = [
    ("none",
     DataAugmentationOptions(auto_contrast=False, invert=False, normalize=False, rotate=False, translate=False,
                             scale=False)),
    ("until_normalize",
     DataAugmentationOptions(auto_contrast=True, invert=True, normalize=True, rotate=False, translate=False,
                             scale=False)),
    ("weak_augmentation",
     DataAugmentationOptions(auto_contrast=True, invert=True, normalize=True, rotate=(-30, 30), translate=(0.15, 0.15),
                             scale=(0.85, 1.15))),
    ("heavy_augmentation",
     DataAugmentationOptions(auto_contrast=True, invert=True, normalize=True, rotate=(-60, 60), translate=(0.3, 0.3),
                             scale=(0.7, 1.3))),
]

# TODO: model and field type dim. r/n only SimpleNet (lg) and command
if __name__ == "__main__":
    trainers = []
    data_arr = []
    for run_config in run_configs:
        for data_split in DataSplit:
            run_id = f"aug_{data_split.value}_{run_config[0]}_cmd"
            print(f'Running for configuration {run_config[0]} and data split {data_split.value}')

            RunUtilsKinderlabor.random_seed()
            # Initialize data loader: data splits and loading from images from disk
            loader_orientation = DataloaderKinderlabor(task_type=TaskType.COMMAND,
                                                       data_split=data_split,
                                                       augmentation_options=run_config[1])

            # visualize class distribution and some (train) samples
            visualizer_orientation = VisualizerKinderlabor(loader_orientation, run_id=run_id)

            # Train model and analyze training progress (mainly when it starts overfitting on validation set)
            trainer_orientation = TrainerKinderlabor(loader_orientation, load_model_from_disk=True, run_id=run_id,
                                                     model_version=ModelVersion.LG)
            trainer_orientation.train_model(n_epochs=50)
            visualizer_orientation.visualize_training_progress(trainer_orientation)

            # Predict on test samples
            trainer_orientation.predict_on_test_samples()
            visualizer_orientation.visualize_model_errors(trainer_orientation)
            trainer_orientation.script_model()

            trainers.append((trainer_orientation, data_split))
            data_arr.append([run_config[0], data_split.value, trainer_orientation.get_predictions()[6]])
    print(tabulate([[x[0], f'{(x[2] * 100):.2f}%', x[1]] for x in data_arr],
                   headers=['Augmentations', 'Performance', 'Data Split']))
    df = pd.DataFrame.from_dict({'augmentation': [row[0] for row in data_arr],
                                 'split': [row[1] for row in data_arr],
                                 'performance': [row[2] for row in data_arr]})
    df.to_csv('./output_visualizations/runs_aug_command.csv', sep=';')
