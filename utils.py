import os
import random
import shutil
from enum import Enum

import numpy as np
import pandas as pd
import torch

from grayscale_model import ModelVersion


class UtilsKinderlabor:
    @staticmethod
    def random_seed():
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

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
    ModelVersion.LG: "simpnet",
    ModelVersion.LE_NET: "lenet",
    ModelVersion.SM: "slim_simpnet"
}

long_names_models = {
    ModelVersion.LG: "SimpleNet",
    ModelVersion.LE_NET: "LeNet-5",
    ModelVersion.SM: "Slimmed SimpleNet"
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
    # DEVANAGARI = "DEVANAGARI"
    EMNIST_LETTERS = "EMNIST_LETTERS"
    MNIST = "MNIST"
    FASHION_MNIST = "FASHION_MNIST"
    ALL_OF_TYPE = "ALL_OF_TYPE"
    FAKE_DATA = "FAKE_DATA"
