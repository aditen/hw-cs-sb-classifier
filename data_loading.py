import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

base_path = "C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/"


class DataloaderKinderlabor:

    def __init__(self, task_type=None, filter_not_readable=True):
        self.__task_type = task_type
        self.__df = pd.read_csv(
            base_path + "20220702_niklas/dataset.csv", sep=";", index_col="id")

        if self.__task_type is not None:
            self.__df = self.__df.loc[(self.__df['type'] == self.__task_type)]
        if filter_not_readable:
            self.__df = self.__df.loc[(self.__df['label'] != "NOT_READABLE")]

        # split train/test randomly
        self.__train_df = self.__df.sample(frac=0.85, random_state=42)
        self.__test_df = self.__df.drop(self.__train_df.index)

        # split train/valid randomly
        self.__valid_df = self.__train_df.sample(frac=0.1, random_state=42)
        self.__train_df = self.__train_df.drop(self.__valid_df.index)

        # create/drop folders and then move samples
        for set_name in ["train_set", "validation_set", "test_set"]:
            if os.path.exists(base_path + set_name):
                shutil.rmtree(base_path + set_name)
            os.mkdir(base_path + set_name)

        for set_name, set_df in [("train_set", self.__train_df),
                                 ("validation_set", self.__valid_df),
                                 ("test_set", self.__test_df)]:
            for idx, row in set_df.iterrows():
                if not os.path.isdir(base_path + set_name + "/" + row['label']):
                    os.mkdir(base_path + set_name + "/" + row['label'])
                shutil.copy(base_path + "20220702_niklas/" + str(idx) + ".jpeg",
                            base_path + set_name + "/" + row[
                                'label'] + "/" + str(
                                idx) + ".jpeg")

        # read image folders and create loaders
        batch_size_train = 16
        batch_size_valid = 8
        batch_size_test = 8
        self.__image_folder_train = ImageFolder(
            base_path + "train_set",
            DataloaderKinderlabor.get_transforms(augment=True, rotate=False))
        self.__dataloader_train = torch.utils.data.DataLoader(self.__image_folder_train, batch_size=batch_size_train,
                                                              shuffle=True, num_workers=min(batch_size_train, 8))
        self.__image_folder_valid = ImageFolder(
            base_path + "validation_set",
            DataloaderKinderlabor.get_transforms(augment=False, rotate=False))
        self.__image_folder_valid.classes = self.__image_folder_train.classes
        self.__image_folder_valid.class_to_idx = self.__image_folder_train.class_to_idx
        self.__dataloader_valid = torch.utils.data.DataLoader(self.__image_folder_valid, batch_size=batch_size_valid,
                                                              shuffle=True, num_workers=min(batch_size_valid, 8))
        self.__image_folder_test = ImageFolder(
            base_path + "test_set",
            DataloaderKinderlabor.get_transforms(augment=False, rotate=False))
        self.__image_folder_test.classes = self.__image_folder_train.classes
        self.__image_folder_test.class_to_idx = self.__image_folder_train.class_to_idx
        self.__dataloader_test = torch.utils.data.DataLoader(self.__image_folder_test, batch_size=batch_size_test,
                                                             shuffle=True, num_workers=min(batch_size_test, 8))

    def plot_class_distributions(self):
        # NOTE: could also plot distributions of train, validation and test sets here
        self.__df.groupby(['type', 'label'] if self.__task_type is None else ['label'])['student'] \
            .count().plot(kind='bar', title='Number of Samples', ylabel='Samples',
                          xlabel='Type and Label', figsize=(6, 5))
        plt.gcf().subplots_adjust(bottom=0.5 if self.__task_type is None else 0.3)
        plt.show()

    def get_num_samples(self):
        return len(self.__image_folder_train), len(self.__image_folder_valid), len(self.__image_folder_test)

    def get_data_loaders(self):
        return self.__dataloader_train, self.__dataloader_valid, self.__dataloader_test

    def get_classes(self):
        return self.__image_folder_train.classes

    def get_task_type(self):
        return self.__task_type

    @staticmethod
    def get_transforms(augment=False, rotate=False):
        if augment:
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomAffine(degrees=(-90, 90) if rotate else (0, 0), translate=(0.15, 0.15),
                                        scale=(0.85, 1.15),
                                        fill=255),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.485], [0.229])
            ])
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
