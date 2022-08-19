import math

import matplotlib.pyplot as plt
import pandas as pd

from data_loading import base_path, dataset_sub_path
from visualizing import class_name_dict

if __name__ == "__main__":
    df = pd.read_csv(
        f'{base_path}{dataset_sub_path}dataset.csv',
        sep=";", index_col="id")
    classes = df['class'].unique()
    print(f'classes currently in df: {classes}')

    task_types = df['type'].unique().tolist()
    print(f'task types: {task_types}')

    demo_class_booklet = "Vishwas Labeling 2"
    demo_class_sheet = 'Data Collection 4. Klasse'

    booklet_df = df[df['class'] == demo_class_booklet]
    sheet_df = df[df['class'] == demo_class_sheet]

    for task_type in task_types:
        if type(task_type) == float and math.isnan(task_type):
            continue
        labels = df[df['type'] == task_type]['label'].unique()
        vals_booklet = [len(booklet_df[booklet_df['label'] == label]) for label in labels]
        vals_sheet = [len(sheet_df[sheet_df['label'] == label]) for label in labels]

        labels = [class_name_dict[x] for x in labels]
        fig, ax = plt.subplots()

        ax.bar(labels, vals_sheet, label='Sheet')
        ax.bar(labels, vals_booklet, label='Booklet', bottom=vals_sheet)

        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Number of Samples per Methodology for type {task_type}')
        ax.legend()

        plt.show()
        fig.savefig(
            f'output_visualizations/methodology_comparison_{task_type}.jpg')
