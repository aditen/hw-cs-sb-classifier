import pandas as pd
from tabulate import tabulate

from data_augmentation import DataAugmentationOptions
from data_loading import DataloaderKinderlabor, TaskType, DataSplit
from grayscale_model import ModelVersion
from run_utils import RunUtilsKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor, data_split_dict, short_names_models

run_configs = [
    ("none", DataAugmentationOptions.none_aug()),
    ("ac", DataAugmentationOptions.auto_ctr_aug()),
    ("eq", DataAugmentationOptions.eq_aug()),
    ("geo", DataAugmentationOptions.geo_aug()),
    ("geo_ac", DataAugmentationOptions.geo_ac_aug()),
    ("crop", DataAugmentationOptions.crop_aug()),
    ("crop_plus", DataAugmentationOptions.crop_plus_aug()),
]

if __name__ == "__main__":
    data_arr = []
    for run_config in run_configs:
        for data_split in DataSplit:
        #for data_split in [DataSplit.HOLD_OUT_CLASSES]:
            for model in [ModelVersion.SM, ModelVersion.LE_NET]:
                run_id = f"baseline_aug[{run_config[0]}]_split[{data_split_dict[data_split]}]_" \
                         f"model[{short_names_models[model]}]"
                print(
                    f'Running for configuration {run_config[0]} and data split {data_split.value} using model {model.name}')

                RunUtilsKinderlabor.random_seed()
                # Initialize data loader: data splits and loading from images from disk
                loader_orientation = DataloaderKinderlabor(task_type=TaskType.COMMAND,
                                                           data_split=data_split,
                                                           augmentation_options=run_config[1])

                # visualize class distribution and some (train) samples
                visualizer_orientation = VisualizerKinderlabor(loader_orientation, run_id=run_id)

                # Train model and analyze training progress (mainly when it starts overfitting on validation set)
                trainer_orientation = TrainerKinderlabor(loader_orientation, load_model_from_disk=True, run_id=run_id,
                                                         model_version=model)
                trainer_orientation.train_model(n_epochs=50)
                visualizer_orientation.visualize_training_progress(trainer_orientation)

                # Predict on test samples
                trainer_orientation.predict_on_test_samples()
                visualizer_orientation.visualize_model_errors(trainer_orientation)
                trainer_orientation.script_model()

                data_arr.append(
                    [run_config[0], data_split_dict[data_split], trainer_orientation.get_predictions()[6], model.name])
    print(tabulate([[x[1], x[3], x[0], f'{(x[2] * 100):.2f}%'] for x in data_arr],
                   headers=['Data Split', 'Model', 'Augmentations', 'Performance']))
    df = pd.DataFrame.from_dict({'augmentation': [row[0] for row in data_arr],
                                 'split': [row[1] for row in data_arr],
                                 'performance': [row[2] for row in data_arr],
                                 'model': [row[3] for row in data_arr]})
    df.to_csv('./output_visualizations/runs_aug_command.csv', sep=';')
