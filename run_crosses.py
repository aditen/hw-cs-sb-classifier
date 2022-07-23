from data_loading import DataloaderKinderlabor
from run_utils import RunUtilsKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor

if __name__ == "__main__":
    # Random seed for reproducibility
    RunUtilsKinderlabor.random_seed()

    # Initialize data loader: data splits and loading from images from disk
    loader_crosses = DataloaderKinderlabor(task_type="CROSS")

    # visualize class distribution and some (train) samples
    visualizer_crosses = VisualizerKinderlabor(loader_crosses)
    visualizer_crosses.plot_class_distributions()
    visualizer_crosses.visualize_some_samples()

    # Train model and analyze training progress (mainly when it starts overfitting on validation set)
    trainer_crosses = TrainerKinderlabor(loader_crosses)
    trainer_crosses.train_model(n_epochs=20)
    visualizer_crosses.visualize_training_progress(trainer_crosses)

    # Predict on test samples
    trainer_crosses.predict_on_test_samples()
    visualizer_crosses.visualize_confusion_matrix(trainer_crosses)
