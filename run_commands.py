from data_loading import DataloaderKinderlabor, TaskType, DataSplit
from run_utils import RunUtilsKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor

if __name__ == "__main__":
    # Random seed for reproducibility
    RunUtilsKinderlabor.random_seed()
    run_id = "lg_cmd_shbl"

    # Initialize data loader: data splits and loading from images from disk
    loader_commands = DataloaderKinderlabor(task_type=TaskType.COMMAND,
                                            data_split=DataSplit.TRAIN_SHEETS_TEST_BOOKLETS)

    # visualize class distribution and some (train) samples
    visualizer_commands = VisualizerKinderlabor(loader_commands, run_id=run_id)
    visualizer_commands.visualize_class_distributions()
    visualizer_commands.visualize_some_samples()

    # Train model and analyze training progress (mainly when it starts overfitting on validation set)
    trainer_commands = TrainerKinderlabor(loader_commands, load_model_from_disk=True, run_id=run_id)
    trainer_commands.train_model(n_epochs=20)
    visualizer_commands.visualize_training_progress(trainer_commands)

    # Predict on test samples
    trainer_commands.predict_on_test_samples()
    visualizer_commands.visualize_model_errors(trainer_commands)
