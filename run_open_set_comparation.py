from data_augmentation import DataAugmentationOptions
from data_loading import DataloaderKinderlabor
from grayscale_model import ModelVersion
from run_utils import Unknowns, TaskType, DataSplit
from training import TrainerKinderlabor, Optimizer
from visualizing import VisualizerKinderlabor

if __name__ == "__main__":
    task_type = TaskType.COMMAND
    for uk_type in [Unknowns.MNIST, Unknowns.EMNIST_LETTERS, Unknowns.FASHION_MNIST]:
        for loss_fc in [Optimizer.ENTROPIC, Optimizer.OBJECTOSPHERE]:
            run_id = f'open_set_task[{task_type.name}]_uk[{uk_type.name}]_loss[{loss_fc.name}]'

            # Initialize data loader: data splits and loading from images from disk
            loader = DataloaderKinderlabor(task_type=task_type,
                                           data_split=DataSplit.HOLD_OUT_CLASSES,
                                           known_unknowns=uk_type,
                                           unknown_unknowns=Unknowns.ALL_OF_TYPE,
                                           augmentation_options=DataAugmentationOptions.geo_ac_aug())

            # visualize class distribution and some (train) samples
            visualizer = VisualizerKinderlabor(loader, run_id=run_id)

            # Train model and analyze training progress (mainly when it starts overfitting on validation set)
            trainer = TrainerKinderlabor(loader, load_model_from_disk=True,
                                         run_id=run_id,
                                         model_version=ModelVersion.SM,
                                         optimizer=loss_fc)
            trainer.train_model(n_epochs=100)
            visualizer.visualize_training_progress(trainer)
            trainer.predict_on_test_samples()
            visualizer.visualize_prob_histogram(trainer)
            visualizer.visualize_model_errors(trainer)
            visualizer.visualize_open_set_recognition_curve(trainer)
