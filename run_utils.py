import os
import random
import shutil

import numpy as np
import pandas as pd
import torch


class RunUtilsKinderlabor:
    @staticmethod
    def random_seed():
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

    @staticmethod
    def copy_to_label_folders(base_origin_folder, base_target_folder, df: pd.DataFrame):
        for idx, row in df.iterrows():
            label_folder = f'{base_target_folder}{row["label"]}/'
            if not os.path.isdir(label_folder):
                os.mkdir(label_folder)
            shutil.copy(f'{base_origin_folder}{str(idx)}.jpeg',
                        f'{label_folder}{str(idx)}.jpeg')
