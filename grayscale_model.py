import torch.nn as nn


# TODO: global average pooling, eventually raise # of channels
# simple CNN from
# https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
class CNN(nn.Module):
    def __init__(self, n_classes: int):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output n classes
        self.out = nn.Linear(32 * 8 * 8, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 8 * 8)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        # no softmax necessary because cross entropy loss does it
        return output
