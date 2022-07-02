import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import torchvision

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(degrees=(-90, 90), translate=(0.25, 0.25), scale=(0.75, 1.25), fill=255),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
    'test': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
}


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

def create_grayscale_datasets(visualize_samples=True, batch_size_train=8):
    image_folder_train = ImageFolder("C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/artificial_dataset",
                                     data_transforms['train'])
    dataloader_train = torch.utils.data.DataLoader(image_folder_train, batch_size=batch_size_train,
                                                   shuffle=True, num_workers=batch_size_train)
    image_folder_test = ImageFolder("C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data",
                                    data_transforms['test'])

    # Use same class indexes in test as in training set
    image_folder_test.class_to_idx = image_folder_train.class_to_idx
    dataloader_test = torch.utils.data.DataLoader(image_folder_test, batch_size=batch_size_train,
                                                  shuffle=True, num_workers=batch_size_train)

    if visualize_samples:
        print("visualizing train images!")
        # Get a batch of training data
        inputs, classes = next(iter(dataloader_train))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        imshow(out, title=[image_folder_train.classes[x] for x in classes])

    return dataloader_train, dataloader_test, len(image_folder_train), len(image_folder_test), image_folder_train.classes
