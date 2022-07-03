import matplotlib.pyplot as plt
import numpy as np
import torchvision

from data_handling.data_loader import DataloaderKinderlabor


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
