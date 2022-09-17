import copy
import math
import os
import shutil
from enum import Enum
from typing import Tuple

import pandas as pd
import torch
import torchvision.datasets
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder

from data_augmentation import DataAugmentationOptions, DataAugmentationUtils


class TaskType(Enum):
    ORIENTATION = "ORIENTATION"
    COMMAND = "COMMAND"
    CROSS = "CROSS"


class DataSplit(Enum):
    TRAIN_SHEETS_TEST_BOOKLETS = "TRAIN_SHEETS_TEST_BOOKLETS"
    HOLD_OUT_CLASSES = "HOLD_OUT_CLASSES"
    RANDOM = "RANDOM"


class Unknowns(Enum):
    DEVANAGARI = "DEVANAGARI"
    EMNIST = "EMNIST"
    FASHION_MNIST = "FASHION_MNIST"


class DataloaderKinderlabor:
    BASE_FOLDER = "C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/"
    SUB_FOLDER = "20220913_field/"
    IMG_CSV_FOLDER = BASE_FOLDER + SUB_FOLDER

    def __init__(self, augmentation_options: DataAugmentationOptions = DataAugmentationOptions(),
                 task_type: TaskType = None, data_split: DataSplit = None, filter_not_readable=True,
                 force_reload_data=False, known_unknowns: Unknowns = None, unknown_unknowns: Unknowns = None,
                 batch_size_train=64, batch_size_valid_test=32):
        self.__augmentation_options = augmentation_options
        self.__task_type = task_type
        self.__data_split = data_split
        self.__force_reload_data = force_reload_data
        self.__known_unknowns = known_unknowns
        self.__unknown_unknowns = unknown_unknowns
        self.__dataset_folder_name = f"{'all' if task_type is None else task_type.value}___" \
                                     f"{DataSplit.RANDOM.value if data_split is None else data_split.value}"
        self.__df = DataloaderKinderlabor.raw_df()
        self.__full_df = self.__df

        self.__mean, self.__std = math.nan, math.nan

        if self.__task_type is not None:
            self.__df = self.__df.loc[(self.__df['type'] == self.__task_type.value)]
        if filter_not_readable:
            self.__df = self.__df.loc[(self.__df['label'] != "NOT_READABLE")]

        if self.__data_split == DataSplit.TRAIN_SHEETS_TEST_BOOKLETS:
            self.__train_df, self.__test_df = self.__split_sheets_booklets(self.__df)
        elif data_split == DataSplit.HOLD_OUT_CLASSES:
            self.__train_df, self.__test_df = self.__split_hold_out(self.__df)
        elif data_split is None or data_split == DataSplit.RANDOM:
            self.__train_df, self.__test_df = self.__split_random(self.__df)
        else:
            raise ValueError(f'Unsupported data split {data_split}')

        # split train/valid randomly
        self.__valid_df = self.__train_df.sample(frac=0.15)
        self.__train_df = self.__train_df.drop(self.__valid_df.index)

        self.__initialize_dataset_folder()

        if isinstance(self.__augmentation_options.normalize, bool) and self.__augmentation_options.normalize is True:
            self.__mean, self.__std = DataAugmentationUtils.determine_mean_std_for_augmentation(
                self.__augmentation_options,
                f'{DataloaderKinderlabor.BASE_FOLDER}{self.__dataset_folder_name}/train_set')
            self.__augmentation_options.normalize = (self.__mean, self.__std)

        if self.__task_type == TaskType.ORIENTATION:
            self.__augmentation_options.rotate = None

        # read image folders and create loaders
        self.__image_folder_train = ImageFolder(
            f'{DataloaderKinderlabor.BASE_FOLDER}{self.__dataset_folder_name}/train_set',
            DataAugmentationUtils.get_augmentations(self.__augmentation_options,
                                                    include_affine=True))
        train_ds = self.__image_folder_train
        if self.__known_unknowns is not None:
            train_ds = self.__add_unknowns_to_df(train_ds, self.__known_unknowns, 2000, -1)
        self.__dataloader_train = DataLoader(train_ds, batch_size=batch_size_train,
                                             shuffle=True, num_workers=0)

        self.__image_folder_valid = ImageFolder(
            f'{DataloaderKinderlabor.BASE_FOLDER}{self.__dataset_folder_name}/validation_set',
            DataAugmentationUtils.get_augmentations(self.__augmentation_options,
                                                    include_affine=False))
        self.__image_folder_valid.classes = self.__image_folder_train.classes
        self.__image_folder_valid.class_to_idx = self.__image_folder_train.class_to_idx
        self.__dataloader_valid = DataLoader(self.__image_folder_valid, batch_size=batch_size_valid_test,
                                             shuffle=False, num_workers=0)

        self.__image_folder_test = ImageFolder(
            f'{DataloaderKinderlabor.BASE_FOLDER}{self.__dataset_folder_name}/test_set',
            DataAugmentationUtils.get_augmentations(self.__augmentation_options,
                                                    include_affine=False))
        self.__image_folder_test.classes = self.__image_folder_train.classes
        self.__image_folder_test.class_to_idx = self.__image_folder_train.class_to_idx
        test_ds = self.__image_folder_test

        if self.__unknown_unknowns is not None:
            test_ds = self.__add_unknowns_to_df(test_ds, self.__unknown_unknowns, 1000, -1)
        self.__dataloader_test = DataLoader(test_ds, batch_size=batch_size_valid_test,
                                            shuffle=False, num_workers=0)

    def get_num_samples(self):
        return len(self.__image_folder_train), len(self.__image_folder_valid), len(self.__image_folder_test)

    def get_data_loaders(self):
        return self.__dataloader_train, self.__dataloader_valid, self.__dataloader_test

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

    def __split_random(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        train_df = df.sample(frac=0.8)
        max_samples_per_class = 1500
        for label in train_df['label'].unique():
            label_df = train_df.loc[train_df['label'] == label]
            if len(label_df) > max_samples_per_class:
                keep_df = label_df.sample(n=max_samples_per_class)
                drop_df = label_df.drop(keep_df.index)
                train_df = train_df.drop(drop_df.index)
        test_df = df.drop(train_df.index)
        return train_df, test_df

    def __split_hold_out(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        test_df = df[(df['class'] == 'Vishwas Labelling 1') | (df['class'] == 'Adrian Labelling 1') | (
                df['class'] == 'Data Collection 4. Klasse')]
        train_df = df.drop(test_df.index)
        return train_df, test_df

    def __split_sheets_booklets(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        if False and self.__task_type == TaskType.COMMAND:
            df = df[
                (df['label'] != 'LOOP_FOUR_TIMES') & (df['label'] != 'LOOP_THREE_TIMES') & (
                        df['label'] != 'LOOP_TWICE') & (df['label'] != 'LOOP_END')]
        train_df = df[
            (df['class'] != 'Vishwas Labelling 1')
            & (df['class'] != 'Vishwas Labeling 2')
            & (df['class'] != 'Adrian Labelling 1')
            & ((df['student'] != 'Laura_Heft_komplett_Test') | (df['label'] == 'EMPTY'))
            ]
        test_df = df.drop(train_df.index)
        return train_df, test_df

    def __initialize_dataset_folder(self):
        if self.__force_reload_data or not os.path.isdir(
                DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name):
            print(f'Creating dataset folder {self.__dataset_folder_name}')
            # create/drop folders and then move samples
            for set_name in ["train_set", "validation_set", "test_set"]:
                if os.path.exists(DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name + "/" + set_name):
                    shutil.rmtree(DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name + "/" + set_name)
                if not os.path.exists(DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name):
                    os.mkdir(DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name)
                os.mkdir(DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name + "/" + set_name)

            for set_name, set_df in [("train_set", self.__train_df),
                                     ("validation_set", self.__valid_df),
                                     ("test_set", self.__test_df)]:
                for idx, row in set_df.iterrows():
                    if not os.path.isdir(
                            DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name + "/" + set_name + "/" + row[
                                'label']):
                        os.mkdir(
                            DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name + "/" + set_name + "/" + row[
                                'label'])
                    shutil.copy(f'{DataloaderKinderlabor.IMG_CSV_FOLDER}{str(idx)}.jpeg',
                                f'{DataloaderKinderlabor.BASE_FOLDER}{self.__dataset_folder_name}/{set_name}/{row["label"]}/{str(idx)}.jpeg')
        else:
            print(f"Skipping dataset folder generation, loading from folder {self.__dataset_folder_name}")

    def __add_unknowns_to_df(self, dataset: Dataset, unknowns: Unknowns, n_to_add: int, unknown_cls_index: int = -1):
        uu_augmentation = copy.deepcopy(self.__augmentation_options)
        uu_augmentation.grayscale = False
        uu_augmentation.invert = False
        if unknowns == Unknowns.EMNIST:
            emnist_set = torchvision.datasets.EMNIST(root="./dataset_root_emnist", split="letters",
                                                     download=True,
                                                     transform=DataAugmentationUtils.get_augmentations(
                                                         uu_augmentation, include_affine=False),
                                                     target_transform=lambda _: unknown_cls_index)
            indices = torch.randperm(len(emnist_set))[:n_to_add]
            emnist_set = Subset(emnist_set, indices)
            return ConcatDataset([dataset, emnist_set])
        elif unknowns == Unknowns.FASHION_MNIST:
            fm_set = torchvision.datasets.FashionMNIST(root="./dataset_root_fashion_mnist", download=True,
                                                       transform=DataAugmentationUtils.get_augmentations(
                                                           uu_augmentation,
                                                           include_affine=False),
                                                       target_transform=lambda _: unknown_cls_index)
            indices = torch.randperm(len(fm_set))[:n_to_add]
            fm_set = Subset(fm_set, indices)
            return ConcatDataset([dataset, fm_set])
        else:
            raise ValueError(f'Unknowns {self.__known_unknowns} not yet supported!')

    @staticmethod
    def raw_df():
        return pd.read_csv(
            f'{DataloaderKinderlabor.IMG_CSV_FOLDER}dataset.csv', sep=";", index_col="id")
