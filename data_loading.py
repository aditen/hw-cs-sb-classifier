import math
import os
import shutil
from enum import Enum

import pandas as pd
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data_augmentation import DataAugmentationOptions, DataAugmentationUtils

base_path = "C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/"
dataset_sub_path = "20220812_more_data/"


class TaskType(Enum):
    ORIENTATION = "ORIENTATION"
    COMMAND = "COMMAND"
    CROSS = "CROSS"


class DataSplit(Enum):
    TRAIN_SHEETS_TEST_BOOKLETS = "TRAIN_SHEETS_TEST_BOOKLETS"
    HOLD_OUT_WITHIN_SHEETS = "HOLD_OUT_WITHIN_SHEETS"


class Unknowns(Enum):
    DEVANAGARI = "DEVANAGARI"
    EMNIST = "EMNIST"
    FASHION_MNIST = "FASHION_MNIST"


class DataloaderKinderlabor:

    def __init__(self, augmentation_options: DataAugmentationOptions = DataAugmentationOptions(),
                 task_type: TaskType = None, data_split: DataSplit = None, filter_not_readable=True,
                 force_reload_data=False, unknown_unknowns: Unknowns = None):
        self.__augmentation_options = augmentation_options
        self.__task_type = task_type
        self.__data_split = data_split
        self.__force_reload_data = force_reload_data
        self.__unknown_unknowns = unknown_unknowns
        self.__dataset_folder_name = f"{'all' if task_type is None else task_type.value}___" \
                                     f"{'all' if data_split is None else data_split.value}"
        self.__df = pd.read_csv(
            f'{base_path}{dataset_sub_path}dataset.csv', sep=";", index_col="id")
        self.__full_df = self.__df

        self.__mean, self.__std = math.nan, math.nan

        if self.__task_type is not None:
            self.__df = self.__df.loc[(self.__df['type'] == self.__task_type.value)]
        if filter_not_readable:
            self.__df = self.__df.loc[(self.__df['label'] != "NOT_READABLE")]

        if self.__data_split is not None:
            if self.__data_split == DataSplit.HOLD_OUT_WITHIN_SHEETS:
                # Test Set based on class 'Data Collection 2. Klasse'
                # split train/test accordingly
                self.__df = self.__df[self.__df['class'] != 'Vishwas Labelling 1']
                self.__df = self.__df[self.__df['class'] != 'Vishwas Labeling 2']
                self.__df = self.__df[self.__df['class'] != 'Adrian Labelling 1']

                self.__train_df = self.__df[self.__df['class'] != 'Data Collection 2. Klasse']
                self.__test_df = self.__df.drop(self.__train_df.index)

                # split train/valid randomly
                self.__valid_df = self.__train_df.sample(frac=0.1, random_state=42)
                self.__train_df = self.__train_df.drop(self.__valid_df.index)
            elif self.__data_split == DataSplit.TRAIN_SHEETS_TEST_BOOKLETS:
                # TODO: analyze different ways to deal with this issue (zero out before softmax)?
                if self.__task_type == TaskType.COMMAND:
                    self.__df = self.__df[
                        (self.__df['label'] != 'LOOP_FOUR_TIMES') & (self.__df['label'] != 'LOOP_THREE_TIMES') & (
                                self.__df['label'] != 'LOOP_TWICE') & (self.__df['label'] != 'LOOP_END')]
                self.__train_df = self.__df[
                    (self.__df['class'] != 'Vishwas Labelling 1')
                    & (self.__df['class'] != 'Vishwas Labeling 2')
                    & (self.__df['class'] != 'Adrian Labelling 1')
                    & ((self.__df['student'] != 'Laura_Heft_komplett_Test') | (self.__df['label'] == 'EMPTY'))
                    ]
                self.__test_df = self.__df.drop(self.__train_df.index)
                # split train/valid randomly
                self.__valid_df = self.__train_df.sample(frac=0.1, random_state=42)
                self.__train_df = self.__train_df.drop(self.__valid_df.index)
            else:
                raise ValueError("Data split is not yet implemented for this value!")
        else:
            # split train/test randomly
            self.__train_df = self.__df.sample(frac=0.85, random_state=42)
            self.__test_df = self.__df.drop(self.__train_df.index)

            # split train/valid randomly
            self.__valid_df = self.__train_df.sample(frac=0.1, random_state=42)
            self.__train_df = self.__train_df.drop(self.__valid_df.index)

        if self.__force_reload_data or not os.path.isdir(base_path + self.__dataset_folder_name):
            print(f'Creating dataset folder {self.__dataset_folder_name}')
            # create/drop folders and then move samples
            for set_name in ["train_set", "validation_set", "test_set"]:
                if os.path.exists(base_path + self.__dataset_folder_name + "/" + set_name):
                    shutil.rmtree(base_path + self.__dataset_folder_name + "/" + set_name)
                if not os.path.exists(base_path + self.__dataset_folder_name):
                    os.mkdir(base_path + self.__dataset_folder_name)
                os.mkdir(base_path + self.__dataset_folder_name + "/" + set_name)

            for set_name, set_df in [("train_set", self.__train_df),
                                     ("validation_set", self.__valid_df),
                                     ("test_set", self.__test_df)]:
                for idx, row in set_df.iterrows():
                    if not os.path.isdir(base_path + self.__dataset_folder_name + "/" + set_name + "/" + row['label']):
                        os.mkdir(base_path + self.__dataset_folder_name + "/" + set_name + "/" + row['label'])
                    shutil.copy(f'{base_path}{dataset_sub_path}{str(idx)}.jpeg',
                                f'{base_path}{self.__dataset_folder_name}/{set_name}/{row["label"]}/{str(idx)}.jpeg')

        else:
            print(f"Skipping dataset folder generation, loading from folder {self.__dataset_folder_name}")

        if isinstance(self.__augmentation_options.normalize, bool) and self.__augmentation_options.normalize is True:
            self.__mean, self.__std = DataAugmentationUtils.determine_mean_std_for_augmentation(
                self.__augmentation_options, f'{base_path}{self.__dataset_folder_name}/train_set')
            self.__augmentation_options.normalize = (self.__mean, self.__std)

        if self.__task_type == TaskType.ORIENTATION:
            self.__augmentation_options.rotate = None

        # read image folders and create loaders
        batch_size_train = 16
        batch_size_valid = 8
        batch_size_test = 8
        self.__image_folder_train = ImageFolder(f'{base_path}{self.__dataset_folder_name}/train_set',
                                                DataAugmentationUtils.get_augmentations(self.__augmentation_options,
                                                                                        include_affine=True))
        self.__dataloader_train = DataLoader(self.__image_folder_train, batch_size=batch_size_train,
                                             shuffle=True, num_workers=min(batch_size_train, 8))
        self.__image_folder_valid = ImageFolder(f'{base_path}{self.__dataset_folder_name}/validation_set',
                                                DataAugmentationUtils.get_augmentations(self.__augmentation_options,
                                                                                        include_affine=False))
        self.__image_folder_valid.classes = self.__image_folder_train.classes
        self.__image_folder_valid.class_to_idx = self.__image_folder_train.class_to_idx
        self.__dataloader_valid = DataLoader(self.__image_folder_valid, batch_size=batch_size_valid,
                                             shuffle=True, num_workers=min(batch_size_valid, 8))
        self.__image_folder_test = ImageFolder(f'{base_path}{self.__dataset_folder_name}/test_set',
                                               DataAugmentationUtils.get_augmentations(self.__augmentation_options,
                                                                                       include_affine=False))
        self.__image_folder_test.classes = self.__image_folder_train.classes
        self.__image_folder_test.class_to_idx = self.__image_folder_train.class_to_idx
        self.__dataloader_test = DataLoader(self.__image_folder_test, batch_size=batch_size_test,
                                            shuffle=True, num_workers=min(batch_size_test, 8))

        if self.__unknown_unknowns is not None:
            if self.__unknown_unknowns == Unknowns.EMNIST:
                self.__augmentation_options.grayscale = False
                self.__augmentation_options.invert = False
                uu_set = torchvision.datasets.EMNIST(root="./dataset_root_emnist", split="letters",
                                                     download=True,
                                                     transform=DataAugmentationUtils.get_augmentations(
                                                         self.__augmentation_options, include_affine=False),
                                                     target_transform=lambda _: -1)
                indices = torch.randperm(len(uu_set))[:3000]
                sampler = torch.utils.data.SubsetRandomSampler(indices)
                self.__uu_loader = DataLoader(uu_set, batch_size=16, sampler=sampler)
            elif self.__unknown_unknowns == Unknowns.FASHION_MNIST:
                self.__augmentation_options.grayscale = False
                self.__augmentation_options.invert = False
                fm_set = torchvision.datasets.FashionMNIST(root="./dataset_root_fashion_mnist", download=True,
                                                           transform=DataAugmentationUtils.get_augmentations(
                                                               self.__augmentation_options,
                                                               include_affine=False),
                                                           target_transform=lambda _: -1)
                self.__uu_loader = DataLoader(fm_set, batch_size=16)

    def get_num_samples(self):
        return len(self.__image_folder_train), len(self.__image_folder_valid), len(self.__image_folder_test)

    def get_data_loaders(self):
        return self.__dataloader_train, self.__dataloader_valid, self.__dataloader_test, self.__uu_loader

    def get_set_dfs(self):
        return self.__train_df, self.__valid_df, self.__test_df, self.__df

    def get_classes(self):
        return self.__image_folder_train.classes

    def get_task_type(self):
        return self.__task_type

    def get_augmentation_options(self):
        return self.__augmentation_options

    def get_mean_std(self):
        return self.__mean, self.__std

    def get_folder_name(self):
        return self.__dataset_folder_name
