import os
import random
import shutil
from enum import Enum

import numpy as np
import pandas as pd
import torch
from torch import nn

from grayscale_model import ModelVersion


class UtilsKinderlabor:
    @staticmethod
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def random_seed(seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)

    @staticmethod
    def copy_to_label_folders(base_origin_folder, base_target_folder, df: pd.DataFrame):
        if not os.path.isdir(base_origin_folder) or not os.path.isdir(base_target_folder):
            raise ValueError('One of the directories is not existing on disk')
        all_labels = df['label'].unique().tolist()
        for label in all_labels:
            label_folder = f'{base_target_folder}{label}/'
            if not os.path.isdir(label_folder):
                os.mkdir(label_folder)
        for idx, row in df.iterrows():
            shutil.copy(f'{base_origin_folder}{str(idx)}.jpeg',
                        f'{base_target_folder}{row["label"]}/{str(idx)}.jpeg')


class_name_dict = {"TURN_RIGHT": "↷", "TURN_LEFT": "↶",
                   "LOOP_THREE_TIMES": "3x", "LOOP_END": "end",
                   "LOOP_TWICE": "2x", "LOOP_FOUR_TIMES": "4x",
                   "PLUS_ONE": "+1", "MINUS_ONE": "-1", "EMPTY": "empty",
                   "ARROW_RIGHT": "→", "ARROW_LEFT": "←", "ARROW_UP": "↑", "ARROW_DOWN": "↓",
                   "CHECKED": "X", "NOT_READABLE": "?", "INSPECT": "*"}


class DataSplit(Enum):
    TRAIN_SHEETS_TEST_BOOKLETS = "TRAIN_SHEETS_TEST_BOOKLETS"
    HOLD_OUT_CLASSES = "HOLD_OUT_CLASSES"
    RANDOM = "RANDOM"


data_split_dict = {DataSplit.TRAIN_SHEETS_TEST_BOOKLETS: "S1", DataSplit.HOLD_OUT_CLASSES: "S2", DataSplit.RANDOM: "S3"}
short_names_models = {
    ModelVersion.LE_NET: "lenet",
    ModelVersion.SM_NO_BOTTLENECK: "slim_nobn",
    ModelVersion.SM_BOTTLENECK: "slim_bn",
    ModelVersion.SM_EOS: "slim_eos",
}


class LossFunction(Enum):
    ENTROPIC = "ENTROPIC"
    OBJECTOSPHERE = "OBJECTOSPHERE"
    SOFTMAX = "SOFTMAX"
    BCE = "BCE"
    ENTROPIC_BCE = "ENTROPIC_BCE"


# NOTE: here one could add 'balanced' early stop that is adapted from paper
#  'Large-Scale Open-Set Classification Protocols for ImageNet'
class EarlyStopCriterion(Enum):
    LOSS = "LOSS"
    BALANCED_ACC = "BALANCED_ACC"


short_names_losses = {
    LossFunction.BCE: "BCE",
    LossFunction.SOFTMAX: "SM",
    LossFunction.ENTROPIC: "EOS",
    LossFunction.ENTROPIC_BCE: "EOS_BIN",
    LossFunction.OBJECTOSPHERE: "OOS"
}

long_names_models = {
    ModelVersion.LE_NET: "LeNet-5",
    ModelVersion.SM_NO_BOTTLENECK: "Slimmed SimpleNet",
    ModelVersion.SM_BOTTLENECK: "Slimmed Simplenet (2D Bottleneck)",
    ModelVersion.SM_EOS: "Slimmed SimpleNet EOS compatible",
}


class TaskType(Enum):
    ORIENTATION = "ORIENTATION"
    COMMAND = "COMMAND"
    CROSS = "CROSS"

short_names_tasks = {
    TaskType.COMMAND: "+1",
    TaskType.ORIENTATION: "^",
    TaskType.CROSS: "x"
}

long_names_tasks = {
    TaskType.COMMAND: "Instruction",
    TaskType.ORIENTATION: "Orientation",
    TaskType.CROSS: "Checkbox"
}


class Unknowns(Enum):
    EMNIST_LETTERS = "EMNIST_LETTERS"
    MNIST = "MNIST"
    FASHION_MNIST = "FASHION_MNIST"
    ALL_OF_TYPE = "ALL_OF_TYPE"
    FAKE_DATA = "FAKE_DATA"
    GAUSSIAN_NOISE_005 = "GAUSSIAN_NOISE_005"
    GAUSSIAN_NOISE_015 = "GAUSSIAN_NOISE_015"
    HOLD_OUT_CLASSES_REST_FAKE_DATA = "HOLD_OUT_CLASSES_REST_FAKE_DATA"


class UnwrapTupleModel(nn.Module):
    def __init__(self, model_returning_tuple: nn.Module):
        super(UnwrapTupleModel, self).__init__()
        self.model = model_returning_tuple

    def forward(self, x):
        out, out_2d = self.model(x)
        return out
