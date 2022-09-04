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

if __name__ == "__main__":
    trainers = []
    for run_config in run_configs:
        run_id = f"aug_{run_config[0]}_cmd"
        print(f'Running for configuration {run_config[0]}')

        RunUtilsKinderlabor.random_seed()
        # Initialize data loader: data splits and loading from images from disk
        loader_orientation = DataloaderKinderlabor(task_type=TaskType.COMMAND,
                                                   data_split=DataSplit.HOLD_OUT_CLASSES,
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

        trainers.append(trainer_orientation)
    print(tabulate([[x[0], f'{(y.get_predictions()[6] * 100):.2f}%'] for x, y in zip(run_configs, trainers)],
                   headers=['Augmentations', 'Performance']))
