from tabulate import tabulate

from data_loading import DataloaderKinderlabor, TaskType, DataSplit
from grayscale_model import SimpleNetVersion, Simplenet
from run_utils import RunUtilsKinderlabor
from training import TrainerKinderlabor
from visualizing import VisualizerKinderlabor


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    size_map = [[version.name, count_parameters(Simplenet(classes=5, version=version))] for version in SimpleNetVersion]
    print(f'Sizes of different models: {size_map}')
    print(tabulate(size_map, headers=["Model Size", "Num Parameters"]))

    RunUtilsKinderlabor.random_seed()

    # Initialize data loader: data splits and loading from images from disk
    loader_orientation = DataloaderKinderlabor(task_type=TaskType.COMMAND,
                                               data_split=DataSplit.TRAIN_SHEETS_TEST_BOOKLETS)

    for version in SimpleNetVersion:
        run_id = f"{version.name.lower()}_cmd_shbl"
        print(f'Running for model size {version.name}')

        # visualize class distribution and some (train) samples
        visualizer_orientation = VisualizerKinderlabor(loader_orientation, run_id=run_id)
        visualizer_orientation.plot_class_distributions()
        visualizer_orientation.visualize_some_samples()

        # Train model and analyze training progress (mainly when it starts overfitting on validation set)
        trainer_orientation = TrainerKinderlabor(loader_orientation, load_model_from_disk=True, run_id=run_id,
                                                 model_version=version)
        trainer_orientation.train_model(n_epochs=12)
        visualizer_orientation.visualize_training_progress(trainer_orientation)

        # Predict on test samples
        trainer_orientation.predict_on_test_samples()
        visualizer_orientation.visualize_model_errors(trainer_orientation)
        trainer_orientation.script_model()
