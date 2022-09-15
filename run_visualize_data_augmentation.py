import math

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data_loading import DataloaderKinderlabor

all_to_visualize = [transforms.RandomAffine(degrees=0), transforms.RandomEqualize(p=1.),
                    transforms.RandomAffine(degrees=(-90, 90)),
                    transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
                    transforms.RandomAffine(degrees=0, scale=(0.5, 1.5))]
titles = ["Original", "Grayscale", "Contrast", "Equalize", "Rotate", "Translate", "Scale"]

all_ids_to_show = [312, 1089, 31382, 34428, 43024, 1299]

if __name__ == "__main__":
    full_df = DataloaderKinderlabor.raw_df()
    filtered_df = pd.DataFrame(columns=full_df.columns)
    for idx in all_ids_to_show:
        filtered_df = pd.concat([filtered_df, full_df[idx:(idx + 1)]])
    raw_imgs = []
    for i, _ in filtered_df.iterrows():
        raw_imgs.append(Image.open(f'{DataloaderKinderlabor.IMG_CSV_FOLDER}{i}.jpeg'))

    n_cols = len(all_to_visualize) + 2
    n_rows = len(all_ids_to_show)
    fig = plt.figure(figsize=(n_cols, n_rows + 1))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.05,
                     share_all=True)

    for i, ax in enumerate(grid):
        raw_img_idx = math.floor(i / n_cols)
        aug_idx = i % n_cols - 2
        raw_img = raw_imgs[raw_img_idx]
        raw_img = transforms.Resize((32, 32))(raw_img)
        if aug_idx > -1:
            all_tfs = transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomAutocontrast(p=1.),
                transforms.RandomInvert(p=1.),
                all_to_visualize[aug_idx]
            ])
            raw_img = all_tfs(raw_img)
        elif aug_idx == -1:
            raw_img = transforms.Compose([
                transforms.RandomInvert(p=1.),
                transforms.Grayscale(),
            ])(raw_img)
        ax.imshow(raw_img, cmap="gray" if aug_idx > -2 else None, vmin=0, vmax=255)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if i < len(titles):
            ax.set_title(titles[i], rotation=90, y=1.1)
    plt.show()
