import pandas as pd
import matplotlib.pyplot as plt

class DataloaderKinderlabor:

    def __init__(self, task_type=None, filter_not_readable=True):
        self.__task_type = task_type
        self.__df = pd.read_csv(
            "C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/20220702_niklas/dataset.csv", sep=";")
        if self.__task_type is not None:
            self.__df = self.__df.loc[(self.__df['type'] == self.__task_type)]
        if filter_not_readable:
            self.__df = self.__df.loc[(self.__df['label'] != "NOT_READABLE")]
        self.__train = self.__df.sample(frac=0.8, random_state=42)
        self.__test = self.__df.drop(self.__train.index)

    def plot_class_distributions(self):
        self.__df.groupby(['type', 'label'] if self.__task_type is None else ['label'])['id'] \
            .count().plot(kind='bar', title='Number of Samples', ylabel='Samples',
                          xlabel='Type and Label', figsize=(6, 5))
        plt.gcf().subplots_adjust(bottom=0.5 if self.__task_type is None else 0.3)
        plt.show()
