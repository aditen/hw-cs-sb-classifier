import math

import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data_augmentation import DataAugmentationOptions, DataAugmentationUtils
from data_loading import DataloaderKinderlabor

vis_augs = [DataAugmentationOptions(to_tensor=False, grayscale=False, invert=False),
            DataAugmentationOptions(to_tensor=False, invert=False),
            DataAugmentationOptions(to_tensor=False),
            DataAugmentationOptions(to_tensor=True, gaussian_noise_sigma=0.025),
            DataAugmentationOptions(to_tensor=True, gaussian_noise_sigma=0.05),
            DataAugmentationOptions(to_tensor=True, gaussian_noise_sigma=0.1),
            DataAugmentationOptions(to_tensor=False, auto_contrast=True),
            DataAugmentationOptions(to_tensor=False, equalize=True),
            DataAugmentationOptions(to_tensor=False, rotate=(-90, 90)),
            DataAugmentationOptions(to_tensor=False, translate=(0.5, 0.5)),
            DataAugmentationOptions(to_tensor=False, scale=(0.5, 1.5)),
            DataAugmentationOptions(to_tensor=False, crop_center=True)]
titles = ["Original", "Grayscale", "Invert", "Noise 1", "Noise 2", "Noise 3", "Contrast", "Equalize", "Rotate",
          "Translate", "Scale", "Crop"]

all_ids_to_show = [312, 1089, 31382, 34428, 43024, 1299]

if __name__ == "__main__":
    raw_imgs = []
    for i in all_ids_to_show:
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
        title = titles[aug_idx]
        if title.startswith("Noise"):
            tf_img = torchvision.transforms.ToPILImage()(tf_img)
        ax.imshow(tf_img, cmap="gray" if aug_idx > 0 else None, vmin=0, vmax=255)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if i < len(titles):
            ax.set_title(titles[i], rotation=90, y=1.1)

    fig.savefig("./output_visualizations/augmentations.pdf")
    plt.show()
