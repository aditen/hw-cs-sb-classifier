import random
import torch

import numpy as np


class RunUtilsKinderlabor:
    @staticmethod
    def random_seed():
        random.seed(42)
        torch.random.manual_seed(42)
        np.random.seed(42)
