import random

import numpy as np
import torch


class RunUtilsKinderlabor:
    @staticmethod
    def random_seed():
        random.seed(42)
        torch.random.manual_seed(42)
        np.random.seed(42)
