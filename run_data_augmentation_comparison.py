from data_augmentation import DataAugmentationOptions
from data_loading import DataloaderKinderlabor, TaskType, DataSplit
from grayscale_model import ModelVersion
from run_utils import RunUtilsKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor

runs_orientation = [(
    "none",
    DataAugmentationOptions(auto_contrast=False, invert=False, normalize=False, rotate=False, translate=False,
                            scale=False)),
    ("autoctr",
     DataAugmentationOptions(auto_contrast=True, invert=False, normalize=False, rotate=False, translate=False,
                             scale=False)),
    ("invert",
     DataAugmentationOptions(auto_contrast=True, invert=True, normalize=False, rotate=False, translate=False,
                             scale=False)),
    ("normalize",
     DataAugmentationOptions(auto_contrast=True, invert=True, normalize=True, rotate=False, translate=False,
                             scale=False)),
]

if __name__ == "__main__":
    for run_config_ort in runs_orientation:
        run_id = f"aug_{run_config_ort[0]}_ort_shbl"
        print(f'Running for configuration {run_config_ort[0]}')

        RunUtilsKinderlabor.random_seed()
        # Initialize data loader: data splits and loading from images from disk
        loader_orientation = DataloaderKinderlabor(task_type=TaskType.ORIENTATION,
                                                   data_split=DataSplit.TRAIN_SHEETS_TEST_BOOKLETS,
                                                   augmentation_options=run_config_ort[1])

        # visualize class distribution and some (train) samples
        visualizer_orientation = VisualizerKinderlabor(loader_orientation, run_id=run_id)

        # Train model and analyze training progress (mainly when it starts overfitting on validation set)
        trainer_orientation = TrainerKinderlabor(loader_orientation, load_model_from_disk=True, run_id=run_id,
                                                 model_version=ModelVersion.SM)
        trainer_orientation.train_model(n_epochs=12)
        visualizer_orientation.visualize_training_progress(trainer_orientation)

        # Predict on test samples
        trainer_orientation.predict_on_test_samples()
        visualizer_orientation.visualize_model_errors(trainer_orientation)
        trainer_orientation.script_model()
