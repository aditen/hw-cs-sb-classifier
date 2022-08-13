from data_loading import DataloaderKinderlabor
from run_utils import RunUtilsKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor

if __name__ == "__main__":
    # Random seed for reproducibility
    RunUtilsKinderlabor.random_seed()

    # Initialize data loader: data splits and loading from images from disk
    data_split = "train_sheets_test_booklets"
    loader_commands = DataloaderKinderlabor(task_type="COMMAND", data_split=data_split)

    # visualize class distribution and some (train) samples
    visualizer_commands = VisualizerKinderlabor(loader_commands)
    visualizer_commands.plot_class_distributions()
    visualizer_commands.visualize_some_samples()

    # Train model and analyze training progress (mainly when it starts overfitting on validation set)
    trainer_commands = TrainerKinderlabor(loader_commands, load_model_from_disk=False, model_path="sn_cmd_shtrbkte.pt")
    trainer_commands.train_model(n_epochs=20)
    visualizer_commands.visualize_training_progress(trainer_commands)

    # Predict on test samples
    trainer_commands.predict_on_test_samples()
    visualizer_commands.visualize_confusion_matrix(trainer_commands)
