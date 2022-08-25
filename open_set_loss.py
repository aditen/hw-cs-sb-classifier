# Copied from vast
import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

_device = None


# TODO: Objectosphere loss from https://github.com/Vastlab/Reducing-Network-Agnostophobia/blob/master/MNIST/Mnist_Training.py

def device(x):
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(_device)


def loss_reducer(func):
    @functools.wraps(func)
    def __loss_reducer__(*args, reduction="none", **kwargs):
        result = func(*args, **kwargs)
        if reduction == "none" or reduction is None:
            return result
        elif reduction == "mean":
            return torch.mean(result)
        elif reduction == "sum":
            return torch.sum(result)

    return __loss_reducer__


class EntropicOpensetLoss:
    def __init__(self, num_of_classes=10):
        self.num_of_classes = num_of_classes
        self.eye = device(torch.eye(self.num_of_classes))
        self.ones = device(torch.ones(self.num_of_classes))
        self.unknowns_multiplier = 1.0 / self.num_of_classes

    @loss_reducer
    def __call__(self, logit_values, target, sample_weights=None):
        catagorical_targets = device(torch.zeros(logit_values.shape))
        known_indexes = target != -1
        unknown_indexes = ~known_indexes
        catagorical_targets[known_indexes, :] = self.eye[target[known_indexes]]
        catagorical_targets[unknown_indexes, :] = (
                self.ones.expand((torch.sum(unknown_indexes).item(), self.num_of_classes))
                * self.unknowns_multiplier
        )
        log_values = F.log_softmax(logit_values, dim=1)
        negative_log_values = -1 * log_values
        loss = negative_log_values * catagorical_targets
        sample_loss = torch.sum(loss, dim=1)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss


# see https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
