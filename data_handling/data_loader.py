import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd

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

        # TODO: read image folders

    def plot_class_distributions(self):
        self.__df.groupby(['type', 'label'] if self.__task_type is None else ['label'])['student'] \
            .count().plot(kind='bar', title='Number of Samples', ylabel='Samples',
                          xlabel='Type and Label', figsize=(6, 5))
        plt.gcf().subplots_adjust(bottom=0.5 if self.__task_type is None else 0.3)
        plt.show()
