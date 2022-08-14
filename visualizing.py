import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from data_loading import DataloaderKinderlabor
from training import TrainerKinderlabor

class_name_dict = {"TURN_RIGHT": "↷", "TURN_LEFT": "↶",
                   "LOOP_THREE_TIMES": "3x", "LOOP_END": "end",
                   "LOOP_TWICE": "2x", "LOOP_FOUR_TIMES": "4x",
                   "PLUS_ONE": "+1", "MINUS_ONE": "-1", "EMPTY": "empty",
                   "ARROW_RIGHT": "→", "ARROW_LEFT": "←", "ARROW_UP": "↑", "ARROW_DOWN": "↓",
                   "CHECKED": "X", "NOT_READABLE": "?"}


def show_on_axis(ax, img_tensor, class_name, mean, std, class_name_predicted=None):
    inp = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([mean])
    std = np.array([std])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = inp.squeeze()
    ax.imshow(inp, cmap='gray_r', vmin=0, vmax=1)
    title = class_name
    if class_name_predicted is not None:
        title = f'pred: {class_name_predicted}, act: {class_name}'
    ax.set_title(title)


class VisualizerKinderlabor:
    def __init__(self, data_loader: DataloaderKinderlabor, save_plots_to_disk=True):
        self.__data_loader = data_loader
        self.__save_plots_to_disk = save_plots_to_disk

    def visualize_some_samples(self):
        train_loader, valid_loader, test_loader = self.__data_loader.get_data_loaders()
        mean, std = self.__data_loader.get_mean_std()
        # Get a batch of training data
        inputs, classes = next(iter(train_loader))

        fig, axs = plt.subplots(4, math.ceil(len(classes) / 4))
        fig.suptitle('Batch of Training Samples', fontsize=16)

        for img_idx in range(len(inputs)):
            show_on_axis(axs.flat[img_idx], inputs[img_idx, :, :],
                         class_name_dict[self.__data_loader.get_classes()[classes[img_idx]]], mean,
                         std)
        fig.tight_layout()
        plt.show()

    def visualize_training_progress(self, trainer: TrainerKinderlabor):
        epochs, train_loss, valid_loss, train_acc, valid_acc = trainer.get_training_progress()
        if len(epochs) > 0:
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, valid_loss, label="Validation Loss")
            plt.plot(epochs, train_acc, label="Training Accuracy")
            plt.plot(epochs, valid_acc, label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Model Performance")
            plt.title("Model Performance over Epochs")
            plt.legend()
            if self.__save_plots_to_disk:
                plt.savefig(
                    f'output_visualizations/learning_progress_{"all" if self.__data_loader.get_task_type() is None else self.__data_loader.get_task_type()}.jpg')
            plt.show()
        else:
            print("No training done yet! Please call this function after training")

    def visualize_confusion_matrix(self, trainer: TrainerKinderlabor):
        actual, predicted, loader = trainer.get_predictions()

        if loader != self.__data_loader:
            print("Loaders are different! Please check you provide the right instance to the visualizer!")
            return

        ConfusionMatrixDisplay.from_predictions(actual, predicted,
                                                display_labels=[class_name_dict[x] for x in loader.get_classes()],
                                                normalize='true')
        if self.__save_plots_to_disk:
            plt.savefig(
                f'output_visualizations/conf_matrix_{"all" if self.__data_loader.get_task_type() is None else self.__data_loader.get_task_type()}.jpg')
        plt.show()

    def plot_class_distributions(self):
        train_df, valid_df, test_df, all_df = self.__data_loader.get_set_dfs()
        labels = all_df['label'].unique()
        vals_train = [len(train_df[train_df['label'] == label]) for label in labels]
        vals_valid = [len(valid_df[valid_df['label'] == label]) for label in labels]
        vals_test = [len(test_df[test_df['label'] == label]) for label in labels]

        labels = [class_name_dict[x] for x in labels]

        fig, ax = plt.subplots()

        ax.bar(labels, vals_train, label='Train Set')
        ax.bar(labels, vals_valid, label='Validation Set', bottom=vals_train)
        ax.bar(labels, vals_test, label='Test Set', bottom=[sum(x) for x in zip(vals_train, vals_valid)])

        ax.set_ylabel('Number of Samples')
        ax.set_title('Number of Samples per Class and Set')
        ax.legend()

        plt.show()
