# Copied from vast
import functools

import torch
from torch.nn import functional as F

_device = None


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
