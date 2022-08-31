import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loading import base_path, dataset_sub_path
from visualizing import class_name_dict

if __name__ == "__main__":
    if not os.path.isdir('output_visualizations/data_availability'):
        os.mkdir('output_visualizations/data_availability')

    df = pd.read_csv(
        f'{base_path}{dataset_sub_path}dataset.csv',
        sep=";", index_col="id")
    classes = df['class'].unique()
    print(f'classes currently in df: {classes}')

    task_types = df['type'].unique().tolist()
    print(f'task types: {task_types}')

    sheet_df = df[(df['class'] != 'Vishwas Labelling 1')
                  & (df['class'] != 'Vishwas Labeling 2')
                  & (df['class'] != 'Adrian Labelling 1')
                  & (df['student'] != 'Laura_Heft_komplett_Test')]
    booklet_df = df.drop(sheet_df.index).loc[(df['student'] != 'Laura_Heft_komplett_Test')]
    booklet_df_single_full = df.loc[(df['student'] == 'Laura_Heft_komplett_Test')]

    for task_type in task_types:
        if type(task_type) == float and math.isnan(task_type):
            continue
        labels = df[df['type'] == task_type]['label'].unique()
        vals_booklet = [len(booklet_df[(booklet_df['label'] == label) & (booklet_df['type'] == task_type)]) for label in labels]
        vals_single_booklet = [len(booklet_df_single_full[(booklet_df_single_full['label'] == label) & (booklet_df_single_full['type'] == task_type)]) for label in labels]
        vals_sheet = [len(sheet_df[(sheet_df['label'] == label) & (sheet_df['type'] == task_type)]) for label in labels]

        labels = [class_name_dict[x] for x in labels]
        fig, ax = plt.subplots()

        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        ax.bar(x - width, vals_sheet, width, label='Sheet')
        ax.bar(x, vals_booklet, width, label='Booklet')
        ax.bar(x + width, vals_single_booklet, width, label='1x Fully Solved Booklet')

        ax.set_ylabel('Number of Samples')
        ax.set_xlabel('Symbol')
        ax.set_title(f'Number of Samples per Methodology for type {task_type}')
        ax.set_xticks(x, labels)
        ax.set_yscale('log')
        ax.legend()

        plt.show()
        fig.savefig(
            f'output_visualizations/data_availability/sample_comp_{task_type}.pdf')
