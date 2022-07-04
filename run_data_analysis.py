from data_handling.data_loader import DataloaderKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor

if __name__ == "__main__":
    loader_orientation = DataloaderKinderlabor(task_type="ORIENTATION")
    loader_orientation.plot_class_distributions()

    visualizer_orientation = VisualizerKinderlabor(loader_orientation)
    visualizer_orientation.visualize_some_samples()

    trainer_orientation = TrainerKinderlabor(loader_orientation)
    trainer_orientation.train_model(n_epochs=10)
    trainer_orientation.visualize_training_progress()
