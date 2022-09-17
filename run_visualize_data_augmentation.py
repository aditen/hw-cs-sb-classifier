import math

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data_augmentation import DataAugmentationOptions, DataAugmentationUtils
from data_loading import DataloaderKinderlabor

vis_augs = [DataAugmentationOptions(to_tensor=False, grayscale=False, invert=False),
            DataAugmentationOptions(to_tensor=False, invert=False),
            DataAugmentationOptions(to_tensor=False, auto_contrast=True),
            DataAugmentationOptions(to_tensor=False, equalize=True),
            DataAugmentationOptions(to_tensor=False, rotate=(-90, 90)),
            DataAugmentationOptions(to_tensor=False, translate=(0.5, 0.5)),
            DataAugmentationOptions(to_tensor=False, scale=(0.5, 1.5)),
            DataAugmentationOptions(to_tensor=False, crop_center=True)]
titles = ["Original", "Grayscale", "Contrast", "Equalize", "Rotate", "Translate", "Scale", "Crop"]

all_ids_to_show = [312, 1089, 31382, 34428, 43024, 1299]

if __name__ == "__main__":
    full_df = DataloaderKinderlabor.raw_df()
    filtered_df = full_df.iloc[all_ids_to_show]
    raw_imgs = []
    for i, _ in filtered_df.iterrows():
        raw_imgs.append(Image.open(f'{DataloaderKinderlabor.IMG_CSV_FOLDER}{i}.jpeg'))

    n_cols = len(vis_augs)
    n_rows = len(all_ids_to_show)
    fig = plt.figure(figsize=(n_cols, n_rows + 1))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.05,
                     share_all=True)

    for i, ax in enumerate(grid):
        raw_img_idx = math.floor(i / n_cols)
        aug_idx = i % n_cols
        raw_img = raw_imgs[raw_img_idx]
        tf_img = DataAugmentationUtils.get_augmentations(vis_augs[aug_idx])(raw_img)
        ax.imshow(tf_img, cmap="gray" if aug_idx > 0 else None, vmin=0, vmax=255)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if i < len(titles):
            ax.set_title(titles[i], rotation=90, y=1.1)

    fig.savefig("./output_visualizations/augmentations.pdf")
    plt.show()