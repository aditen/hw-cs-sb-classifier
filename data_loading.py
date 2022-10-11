import copy
import math
import os
import shutil

import pandas as pd
import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder

from data_augmentation import DataAugmentationOptions, DataAugmentationUtils
from utils import UtilsKinderlabor, data_split_dict, TaskType, DataSplit, Unknowns


class DataloaderKinderlabor:
    BASE_FOLDER = "./kinderlabor_dataset/"
    IMG_CSV_FOLDER = BASE_FOLDER

    def __init__(self, augmentation_options: DataAugmentationOptions = DataAugmentationOptions(),
                 task_type: TaskType = None, data_split: DataSplit = None, filter_not_readable=True,
                 force_reload_data=False, known_unknowns: Unknowns = None, unknown_unknowns: Unknowns = None,
                 batch_size_train=64, batch_size_valid_test=32):
        UtilsKinderlabor.random_seed()
        self.__augmentation_options = augmentation_options
        self.__task_type = task_type
        self.__data_split = data_split
        self.__force_reload_data = force_reload_data
        self.__known_unknowns = known_unknowns
        self.__unknown_unknowns = unknown_unknowns
        self.__dataset_folder_name = f"{'all' if task_type is None else task_type.value}___" \
                                     f"{DataSplit.RANDOM.value if data_split is None else data_split.value}"
        self.__df = DataloaderKinderlabor.full_anonymized_df()
        self.__full_df = self.__df

        self.__mean, self.__std = math.nan, math.nan

        if self.__task_type is not None:
            self.__df = self.__df.loc[(self.__df['type'] == self.__task_type.value)]
        if filter_not_readable:
            self.__df = self.__df.loc[(self.__df['label'] != "NOT_READABLE")]

        if data_split_dict[data_split] not in self.__df.columns:
            raise ValueError(f'Unsupported data split {data_split}')
        self.__train_df = self.__df[self.__df[data_split_dict[data_split]] == "train"]
        self.__valid_df = self.__df[self.__df[data_split_dict[data_split]] == "valid"]
        self.__test_df = self.__df[self.__df[data_split_dict[data_split]] == "test"]

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
            train_ds = self.__add_unknowns_to_df(train_ds, self.__known_unknowns, 1700, -1)
        self.__dataloader_train = DataLoader(train_ds, batch_size=batch_size_train,
                                             shuffle=True, num_workers=0)

        self.__image_folder_valid = ImageFolder(
            f'{DataloaderKinderlabor.BASE_FOLDER}{self.__dataset_folder_name}/validation_set',
            DataAugmentationUtils.get_augmentations(self.__augmentation_options,
                                                    include_affine=False))
        self.__image_folder_valid.classes = self.__image_folder_train.classes
        self.__image_folder_valid.class_to_idx = self.__image_folder_train.class_to_idx
        valid_ds = self.__image_folder_valid
        if self.__known_unknowns is not None:
            valid_ds = self.__add_unknowns_to_df(valid_ds, self.__known_unknowns, 300, -1)
        self.__dataloader_valid = DataLoader(valid_ds, batch_size=batch_size_valid_test,
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

    def __initialize_dataset_folder(self):
        if self.__force_reload_data or not os.path.isdir(
                DataloaderKinderlabor.BASE_FOLDER + self.__dataset_folder_name):
            print(f'Creating dataset folder {self.__dataset_folder_name}')
            # create/drop folders and then move samples
            for set_name in ["train_set", "validation_set", "test_set"]:
                set_folder = f'{DataloaderKinderlabor.BASE_FOLDER}{self.__dataset_folder_name}/{set_name}'
                if os.path.exists(set_folder):
                    shutil.rmtree(set_folder)
                os.makedirs(set_folder)

            for set_name, set_df in [("train_set", self.__train_df),
                                     ("validation_set", self.__valid_df),
                                     ("test_set", self.__test_df)]:
                UtilsKinderlabor.copy_to_label_folders(base_origin_folder=DataloaderKinderlabor.IMG_CSV_FOLDER,
                                                       base_target_folder=f'{DataloaderKinderlabor.BASE_FOLDER}{self.__dataset_folder_name}/{set_name}/',
                                                       df=set_df)
        else:
            print(f"Skipping dataset folder generation, loading from folder {self.__dataset_folder_name}")

    def __add_unknowns_to_df(self, dataset: Dataset, unknowns: Unknowns, n_to_add: int, unknown_cls_index: int = -1):
        uu_augmentation = copy.deepcopy(self.__augmentation_options)
        uu_augmentation.grayscale = False
        uu_augmentation.invert = False
        if unknowns == Unknowns.EMNIST_LETTERS:
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
        elif unknowns == Unknowns.MNIST:
            fm_set = torchvision.datasets.MNIST(root="./dataset_root_mnist", download=True,
                                                transform=DataAugmentationUtils.get_augmentations(
                                                    uu_augmentation,
                                                    include_affine=False),
                                                target_transform=lambda _: unknown_cls_index)
            indices = torch.randperm(len(fm_set))[:n_to_add]
            fm_set = Subset(fm_set, indices)
            return ConcatDataset([dataset, fm_set])
        elif unknowns == Unknowns.FAKE_DATA:
            UtilsKinderlabor.random_seed()
            fd_set = torchvision.datasets.FakeData(size=n_to_add, image_size=(1, 32, 32),
                                                   transform=DataAugmentationUtils.get_augmentations(
                                                       uu_augmentation,
                                                       include_affine=False),
                                                   target_transform=lambda _: torch.tensor(unknown_cls_index))
            return ConcatDataset([dataset, fd_set])
        elif unknowns == Unknowns.ALL_OF_TYPE:
            if self.__task_type is None:
                raise ValueError("Unknowns of type needs a task type defined")
            uk_df = self.__full_df[
                (self.__full_df['label'] == 'NOT_READABLE') & (self.__full_df['type'] == self.__task_type.value)]
            uk_dir = f'./kinderlabor_dataset/unknowns_{self.__task_type.value}/'
            if not os.path.isdir(uk_dir):
                os.makedirs(uk_dir)
                UtilsKinderlabor.copy_to_label_folders(base_origin_folder=DataloaderKinderlabor.IMG_CSV_FOLDER,
                                                       base_target_folder=uk_dir, df=uk_df)
            img_folder = ImageFolder(
                uk_dir, DataAugmentationUtils.get_augmentations(self.__augmentation_options,
                                                                include_affine=False),
                target_transform=lambda _: unknown_cls_index)
            return ConcatDataset([dataset, img_folder])
        elif unknowns == Unknowns.GAUSSIAN_NOISE_03 or unknowns == Unknowns.GAUSSIAN_NOISE_015:
            if not isinstance(dataset, ImageFolder):
                raise ValueError("Cannot use original data set for gaussian noise as it is no image folder!")
            aug_uk = DataAugmentationOptions(
                gaussian_noise_sigma=0.15 if unknowns == Unknowns.GAUSSIAN_NOISE_015 else 0.3,
                auto_contrast=self.__augmentation_options.auto_contrast)
            img_folder_uk = ImageFolder(dataset.root,
                                        transform=DataAugmentationUtils.get_augmentations(aug_uk, include_affine=False),
                                        target_transform=lambda y: unknown_cls_index)
            indices = torch.randperm(len(img_folder_uk))[:n_to_add]
            img_folder_uk = Subset(img_folder_uk, indices)
            return ConcatDataset([dataset, img_folder_uk])
        else:
            raise ValueError(f'Unknowns {self.__known_unknowns} not yet supported!')

    @staticmethod
    def full_anonymized_df(include_inspects=False):
        df = pd.read_csv(
            f'./kinderlabor_dataset/dataset_anonymized.csv', sep=";")
        if include_inspects is not True:
            df = df[df['label'] != 'INSPECT']
        return df

    @staticmethod
    def raw_herby_df():
        df = pd.read_csv(
            'C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/20220925_corr_v2/dataset.csv',
            sep=";")
        # filter exercise that was different in print than in Herby version
        df = df[(df['exercise'] != '12e') & (df['exercise'] != '12f')]
        # filter Kinderlabor 4 because drawing fields there are no type, just some random unknowns basically
        df = df[df['sheet'] != 'Kinderlabor 4']
        # filter class that was experimentally "self-labelling" (but leave data to observe this different aspect)
        df = df[df['class'] != 'Trimmis 3 / 4']
        # filter exercises with no type (path drawing, unknowns in other context)
        df = df[df['type'].notnull()]
        return df
