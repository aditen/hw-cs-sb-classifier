import os
import shutil
from typing import Tuple, List

from pandas import DataFrame

from data_loading import DataloaderKinderlabor
from utils import UtilsKinderlabor, data_split_dict, TaskType, DataSplit


def __split_random(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
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


def __split_hold_out(df: DataFrame, hold_out: List[str]) -> Tuple[DataFrame, DataFrame]:
    test_df = df[df['class'].isin(hold_out)]
    train_df = df.drop(test_df.index)
    return train_df, test_df


def __split_sheets_booklets(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    train_df = df[
        (df['sheet'] == 'Datensammelblatt Kinderlabor') |
        (df['sheet'] == 'Data Collection 1. Klasse') |
        ((df['student'] == 'Laura_Heft_komplett_Test') & (df['label'] == 'EMPTY'))
        ]
    test_df = df.drop(train_df.index)
    return train_df, test_df


# creates anonymous dataset from (pseudonymous) herby output (pseudonymous to still have class/student association)
if __name__ == "__main__":
    UtilsKinderlabor.random_seed()
    full_df = DataloaderKinderlabor.raw_herby_df()
    hold_out_classes = os.getenv('HOLDOUTCLASSES', default=None)
    if hold_out_classes is None:
        raise ValueError('Hold Out classes env variable not defined!')
    hold_out_classes = hold_out_classes.split(",")
    print(f"Hold out classes: {hold_out_classes}")

    kwargs_split_cols = {data_split_dict[val]: "" for val in DataSplit}
    full_df = full_df.assign(**kwargs_split_cols)

    # INSPECT does not belong to any splits!
    df = full_df[full_df['label'] != 'INSPECT']

    for split in DataSplit:
        for task_type in TaskType:
            task_df = df[df['type'] == task_type.value]
            if split == DataSplit.RANDOM:
                train, test = __split_random(task_df)
            elif split == DataSplit.HOLD_OUT_CLASSES:
                train, test = __split_hold_out(task_df, hold_out=hold_out_classes)
            elif split == DataSplit.TRAIN_SHEETS_TEST_BOOKLETS:
                train, test = __split_sheets_booklets(task_df)
            else:
                raise ValueError()
            valid = train.sample(frac=0.15)
            train = train.drop(valid.index)
            full_df.loc[train.index, data_split_dict[split]] = "train"
            full_df.loc[valid.index, data_split_dict[split]] = "valid"
            full_df.loc[test.index, data_split_dict[split]] = "test"

    full_df = full_df.drop(['request', 'student', 'class', 'sheet', 'field', 'exercise'], axis=1)
    full_df.to_csv('./kinderlabor_dataset/dataset_anonymized.csv', sep=";")

    for idx, row in full_df.iterrows():
        shutil.copy(f'C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/20220925_corr_v2/{idx}.jpeg', f'./kinderlabor_dataset/{row["id"]}.jpeg')
