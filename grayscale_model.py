from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelVersion(Enum):
    SM_BOTTLENECK = "SM_BOTTLENECK"
    SM_NO_BOTTLENECK = "SM_NO_BOTTLENECK"
    LE_NET = "LE_NET"


def get_model(model_version: ModelVersion, num_classes: int):
    num_classes = 1 if num_classes == 2 else num_classes
    if model_version == ModelVersion.SM_NO_BOTTLENECK:
        return SimpleNetSlimmed(classes=num_classes)
    if model_version == ModelVersion.LE_NET:
        return LeNet(n_classes=num_classes)
    elif model_version == ModelVersion.SM_BOTTLENECK:
        return SimpleNetSlimmed2D(classes=num_classes)
    else:
        raise ValueError('Unsupported model version supplied')


# adapted to n_classes
# from PyTorch Tutorial: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class LeNet(nn.Module):

    def __init__(self, n_classes=10):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # we do not do any Space visualization here!
        return x, None


'''
SimpleNetV1 in Pytorch.
The implementation is from: 
https://github.com/Coderx7/SimpleNet_Pytorch
'''


def _make_simplenet_layers(in_channels=1, channel_divisor=1):
    n_channels_to_int = int(64 / channel_divisor)
    model = nn.Sequential(
        nn.Conv2d(in_channels, n_channels_to_int, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(n_channels_to_int, n_channels_to_int * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 2, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(n_channels_to_int * 2, n_channels_to_int * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 2, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(n_channels_to_int * 2, n_channels_to_int * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 2, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(n_channels_to_int * 2, n_channels_to_int * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 2, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(n_channels_to_int * 2, n_channels_to_int * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 2, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(n_channels_to_int * 2, n_channels_to_int * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 4, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(n_channels_to_int * 4, n_channels_to_int * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 4, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(n_channels_to_int * 4, n_channels_to_int * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 4, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(n_channels_to_int * 4, n_channels_to_int * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 8, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(n_channels_to_int * 8, n_channels_to_int * 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        nn.BatchNorm2d(n_channels_to_int * 32, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(n_channels_to_int * 32, n_channels_to_int * 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        nn.BatchNorm2d(n_channels_to_int * 4, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(n_channels_to_int * 4, n_channels_to_int * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(n_channels_to_int * 4, eps=1e-05, momentum=0.05, affine=True),
        nn.ReLU(inplace=True),
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    return model


class SimpleNetSlimmed2D(nn.Module):
    def __init__(self, classes=10, in_channels=1):
        super(SimpleNetSlimmed2D, self).__init__()
        self.features = _make_simplenet_layers(in_channels=in_channels, channel_divisor=4)
        self.to2D = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )
        self.after2D = nn.Linear(2, classes, bias=False)

    def forward(self, x):
        features = self.features(x)
        # Global Max Pooling
        features = F.max_pool2d(features, kernel_size=features.size()[2:])
        out_2d = self.to2D(features)
        out = self.after2D(out_2d)
        return out, out_2d


class SimpleNetSlimmed:
    def __init__(self, classes=10, in_channels=1):
        super(SimpleNetSlimmed, self).__init__()
        self.features = _make_simplenet_layers(in_channels=in_channels, channel_divisor=4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        features = self.features(x)
        # Global Max Pooling
        features = F.max_pool2d(features, kernel_size=features.size()[2:])
        out = self.classifier(features)
        return out, None
