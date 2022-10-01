from data_augmentation import DataAugmentationOptions
from data_loading import DataloaderKinderlabor
from grayscale_model import ModelVersion
from run_baseline import get_run_id
from run_utils import TaskType, DataSplit, Unknowns
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor

if __name__ == "__main__":
    for task_type in TaskType:
        run_id = get_run_id(task_type, "geo_ac", DataSplit.HOLD_OUT_CLASSES, ModelVersion.SM)

        # Initialize data loader: data splits and loading from images from disk
        loader = DataloaderKinderlabor(task_type=task_type,
                                       data_split=DataSplit.HOLD_OUT_CLASSES,
                                       unknown_unknowns=Unknowns.ALL_OF_TYPE,
                                       augmentation_options=DataAugmentationOptions.geo_ac_aug())

        # visualize class distribution and some (train) samples
        visualizer = VisualizerKinderlabor(loader, run_id=f"{run_id}_open_set")

        # Train model and analyze training progress (mainly when it starts overfitting on validation set)
        trainer = TrainerKinderlabor(loader, load_model_from_disk=True,
                                     run_id=run_id,
                                     model_version=ModelVersion.SM)
        trainer.train_model(n_epochs=75)
        visualizer.visualize_training_progress(trainer)
        trainer.predict_on_test_samples()
        visualizer.visualize_prob_histogram(trainer)
        visualizer.visualize_model_errors(trainer)
        visualizer.visualize_open_set_recognition_curve(trainer)
