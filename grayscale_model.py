from enum import Enum

import torch.nn as nn
import torch.nn.functional as F


class ModelVersion(Enum):
    XS = 8
    SM = 4
    MD = 2
    LG = 1
    LE_NET_PP = 0


def get_model(model_version: ModelVersion, num_classes):
    if model_version == ModelVersion.LE_NET_PP:
        return LeNetPP(num_classes)
    return SimpleNet(version=model_version, classes=num_classes)


# LeNet++ from Center Loss Paper, see https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/models.py
class LeNetPP(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, num_classes):
        super(LeNetPP, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 4 * 4, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 4 * 4)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        # TODO: enable later for plotting
        # return x, y
        return y


'''
SimplerNetV1 in Pytorch.
The implementation is from: 
https://github.com/Coderx7/SimpleNet_Pytorch
'''


# Note: LG = default
class SimpleNet(nn.Module):
    def __init__(self, classes=10, in_channels=1, version: ModelVersion = ModelVersion.LG):
        super(SimpleNet, self).__init__()
        self.features = self._make_layers(in_channels=in_channels, channel_divisor=version.value)
        n_channels_to_int = int(64 / version.value)
        self.classifier = nn.Linear(n_channels_to_int * 4, classes)
        self.drp = nn.Dropout(0.1)

    def forward(self, x):
        out = self.features(x)

        # Global Max Pooling (if input is >32x32, in our case already is 1x1)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        # dropout, as it is 1x1 already no 2D Dropout is necessary
        out = self.drp(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, in_channels=1, channel_divisor=1):

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
