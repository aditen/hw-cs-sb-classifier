import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay
from data_handling.data_loader import DataloaderKinderlabor
from training import TrainerKinderlabor


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485])
    std = np.array([0.229])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class VisualizerKinderlabor:
    def __init__(self, data_loader: DataloaderKinderlabor):
        self.__data_loader = data_loader

    def visualize_some_samples(self):
        train_loader, valid_loader, test_loader = self.__data_loader.get_data_loaders()
        # Get a batch of training data
        inputs, classes = next(iter(train_loader))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        imshow(out, title=[self.__data_loader.get_classes()[x] for x in classes])

    def visualize_confusion_matrix(self, trainer: TrainerKinderlabor):
        actual, predicted, loader = trainer.get_predictions()
        class_transforms = {"ARROW_DOWN": "down", "ARROW_LEFT": "left", "ARROW_RIGHT": "right", "ARROW_UP": "up",
                            "EMPTY": "empty"}
        if loader != self.__data_loader:
            print("Loaders are different! Please check you provide the right instance to the visualizer!")
            return
        ConfusionMatrixDisplay.from_predictions(actual, predicted,
                                                display_labels=[class_transforms[x] for x in loader.get_classes()])
        plt.show()
