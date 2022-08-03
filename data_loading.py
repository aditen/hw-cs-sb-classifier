import math
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

base_path = "C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/"
dataset_sub_path = "20220803_over_border_cropping/"


class DataloaderKinderlabor:

    def __init__(self, task_type=None, filter_not_readable=True, data_split=None):
        self.__task_type = task_type
        self.__data_split = data_split
        self.__df = pd.read_csv(
            f'{base_path}{dataset_sub_path}dataset.csv', sep=";", index_col="id")
        self.__full_df = self.__df

        if self.__task_type is not None:
            self.__df = self.__df.loc[(self.__df['type'] == self.__task_type)]
        if filter_not_readable:
            self.__df = self.__df.loc[(self.__df['label'] != "NOT_READABLE")]

        if self.__data_split is not None:
            if self.__data_split == "hold_out_2nd":
                # Test Set based on class 'Data Collection 2. Klasse'
                # split train/test accordingly
                self.__df = self.__df[self.__df['class'] != 'Vishwas Labelling 1']
                self.__df = self.__df[self.__df['class'] != 'Vishwas Labeling 2']

                self.__train_df = self.__df[self.__df['class'] != 'Data Collection 2. Klasse']
                self.__test_df = self.__df.drop(self.__train_df.index)

                # split train/valid randomly
                self.__valid_df = self.__train_df.sample(frac=0.1, random_state=42)
                self.__train_df = self.__train_df.drop(self.__valid_df.index)
            elif self.__data_split == "train_sheets_test_booklets":
                # TODO: remove again and use mask or similar, but we hardly have any loops in the test set as of now
                if self.__task_type == "COMMAND":
                    self.__df = self.__df[
                        (self.__df['label'] != 'LOOP_FOUR_TIMES') & (self.__df['label'] != 'LOOP_THREE_TIMES') & (
                                self.__df['label'] != 'LOOP_TWICE') & (self.__df['label'] != 'LOOP_END')]
                self.__train_df = self.__df[
                    (self.__df['class'] != 'Vishwas Labelling 1')
                    #& (self.__df['class'] != 'Vishwas Labeling 2')
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

        # create/drop folders and then move samples
        for set_name in ["train_set", "validation_set", "test_set"]:
            if os.path.exists(base_path + self.__task_type + "/" + set_name):
                shutil.rmtree(base_path + self.__task_type + "/" + set_name)
            if not os.path.exists(base_path + self.__task_type):
                os.mkdir(base_path + self.__task_type)
            os.mkdir(base_path + self.__task_type + "/" + set_name)

        for set_name, set_df in [("train_set", self.__train_df),
                                 ("validation_set", self.__valid_df),
                                 ("test_set", self.__test_df)]:
            for idx, row in set_df.iterrows():
                if not os.path.isdir(base_path + self.__task_type + "/" + set_name + "/" + row['label']):
                    os.mkdir(base_path + self.__task_type + "/" + set_name + "/" + row['label'])
                shutil.copy(f'{base_path}{dataset_sub_path}{str(idx)}.jpeg',
                            f'{base_path}{self.__task_type}/{set_name}/{row["label"]}/{str(idx)}.jpeg')

        # calculate mean and std on dataset
        transforms_get_mean_std = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.RandomAutocontrast(p=1.),
            transforms.RandomInvert(p=1.),
            transforms.ToTensor()
        ])
        self.__image_folder_std_mean = ImageFolder(f'{base_path}{self.__task_type}/train_set',
                                                   transforms_get_mean_std)
        self.__dataloader_std_mean = torch.utils.data.DataLoader(self.__image_folder_std_mean,
                                                                 batch_size=8,
                                                                 shuffle=True, num_workers=8)

        mean_sum = 0.
        n_total = 0
        var_sum = 0.

        for imgs, _ in self.__dataloader_std_mean:
            for img in imgs:
                mean_sum += img.mean().item()
                n_total += 1
        self.__mean = mean_sum / n_total

        for imgs, _ in self.__dataloader_std_mean:
            for img in imgs:
                stds = ((img - self.__mean) ** 2)
                var = stds.sum().item() / (32 * 32)
                var_sum += var

        self.__std = math.sqrt(var_sum / n_total)
        print(f'Dataset mean: {self.__mean:.4f}, std: {self.__std:.4f}')

        # read image folders and create loaders
        batch_size_train = 16
        batch_size_valid = 8
        batch_size_test = 8
        self.__image_folder_train = ImageFolder(f'{base_path}{self.__task_type}/train_set',
                                                self.get_transforms(augment=True,
                                                                    rotate=self.__task_type != "ORIENTATION"))
        self.__dataloader_train = torch.utils.data.DataLoader(self.__image_folder_train, batch_size=batch_size_train,
                                                              shuffle=True, num_workers=min(batch_size_train, 8))
        self.__image_folder_valid = ImageFolder(f'{base_path}{self.__task_type}/validation_set',
                                                self.get_transforms(augment=False, rotate=False))
        self.__image_folder_valid.classes = self.__image_folder_train.classes
        self.__image_folder_valid.class_to_idx = self.__image_folder_train.class_to_idx
        self.__dataloader_valid = torch.utils.data.DataLoader(self.__image_folder_valid, batch_size=batch_size_valid,
                                                              shuffle=True, num_workers=min(batch_size_valid, 8))
        self.__image_folder_test = ImageFolder(f'{base_path}{self.__task_type}/test_set',
                                               self.get_transforms(augment=False, rotate=False))
        self.__image_folder_test.classes = self.__image_folder_train.classes
        self.__image_folder_test.class_to_idx = self.__image_folder_train.class_to_idx
        self.__dataloader_test = torch.utils.data.DataLoader(self.__image_folder_test, batch_size=batch_size_test,
                                                             shuffle=True, num_workers=min(batch_size_test, 8))

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

    # TODO: change to 28x28 if using MNIST for pre-training/unknowns
    # TODO: cut over borders in Herby, random Crop down (e.g. 36x36 to 28x28)
    def get_transforms(self, augment=False, rotate=False):
        if augment:
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(),
                transforms.RandomAutocontrast(p=1.),
                transforms.RandomInvert(p=1.),
                transforms.RandomAffine(degrees=(-30, 30) if rotate else (0, 0), translate=(0.15, 0.15),
                                        scale=(0.85, 1.15)),
                transforms.ToTensor(),
                transforms.Normalize([self.__mean], [self.__std])
            ])
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.RandomAutocontrast(p=1.),
            transforms.RandomInvert(p=1.),
            transforms.ToTensor(),
            transforms.Normalize([self.__mean], [self.__std])
        ])

    def get_mean_std(self):
        return self.__mean, self.__std
