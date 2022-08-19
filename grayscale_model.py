from enum import Enum

'''
SimplerNetV1 in Pytorch.
The implementation is from: 
https://github.com/Coderx7/SimpleNet_Pytorch
'''
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetVersion(Enum):
    XS = 8
    SM = 4
    MD = 2
    LG = 1


# Note: LG = default
class Simplenet(nn.Module):
    def __init__(self, classes=10, in_channels=1, version: SimpleNetVersion = SimpleNetVersion.LG):
        super(Simplenet, self).__init__()
        self.features = self._make_layers(in_channels=in_channels, channel_divisor=version.value)
        n_channels_to_int = int(64 / version.value)
        self.classifier = nn.Linear(n_channels_to_int * 4, classes)
        self.drp = nn.Dropout(0.1)

    def forward(self, x):
        out = self.features(x)

        # Global Max Pooling (if input is >32x32, else already is 1x1)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        # out = F.dropout2d(out, 0.1, training=True)
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
