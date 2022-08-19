from data_loading import DataloaderKinderlabor, TaskType, DataSplit
from run_utils import RunUtilsKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor

if __name__ == "__main__":
    # Random seed for reproducibility
    RunUtilsKinderlabor.random_seed()
    run_id = "lg_ort_shbl"

    # Initialize data loader: data splits and loading from images from disk
    loader_orientation = DataloaderKinderlabor(task_type=TaskType.ORIENTATION,
                                               data_split=DataSplit.TRAIN_SHEETS_TEST_BOOKLETS)

    # visualize class distribution and some (train) samples
    visualizer_orientation = VisualizerKinderlabor(loader_orientation, run_id=run_id)
    visualizer_orientation.plot_class_distributions()
    visualizer_orientation.visualize_some_samples()

    # Train model and analyze training progress (mainly when it starts overfitting on validation set)
    trainer_orientation = TrainerKinderlabor(loader_orientation, load_model_from_disk=False, run_id=run_id)
    trainer_orientation.train_model(n_epochs=10)
    visualizer_orientation.visualize_training_progress(trainer_orientation)

    # Predict on test samples
    trainer_orientation.predict_on_test_samples()
    visualizer_orientation.visualize_model_errors(trainer_orientation)
    trainer_orientation.script_model()
