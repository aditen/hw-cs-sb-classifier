from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


@dataclass
class DataAugmentationOptions:
    auto_contrast: bool = True
    invert: bool = True
    normalize: Tuple[float] | bool = True
    rotate: int | Tuple[int] | bool = (-30, 30)
    translate: float | Tuple[float] | bool = (0.15, 0.15)
    scale: float | Tuple[float] | bool = (0.85, 1.15)


class DataAugmentationUtils:

    @staticmethod
    def __get_transforms(options: DataAugmentationOptions):
        transforms_list = [transforms.ToTensor(),
                           transforms.Grayscale(),
                           transforms.Resize((32, 32))]

        # Auto Contrast is yes or no
        if options.auto_contrast:
            transforms_list.append(transforms.RandomAutocontrast(p=1.))

        # Inverting is yes or no
        if options.invert:
            transforms_list.append(transforms.RandomInvert(p=1.))

        # Rotation, translation or scaling can be
        if options.rotate or options.translate or options.scale:
            kwargs = {'degrees': (0, 0)}
            # Rotation degrees
            if isinstance(options.rotate, tuple):
                kwargs['degrees'] = options.rotate
            elif isinstance(options.rotate, int):
                kwargs['degrees'] = (-options.rotate, options.rotate)
            elif isinstance(options.rotate, bool) and options.rotate is True:
                kwargs['degrees'] = (-30, 30)

            # Translation values (relative)
            if isinstance(options.translate, tuple):
                kwargs['translate'] = options.translate
            elif isinstance(options.translate, float):
                kwargs['translate'] = (options.translate, options.translate)
            elif isinstance(options.translate, bool) and options.translate is True:
                kwargs['translate'] = (0.15, 0.15)

            # Scaling values (relative)
            if isinstance(options.scale, tuple):
                kwargs['scale'] = options.scale
            elif isinstance(options.scale, float):
                kwargs['scale'] = (1. - options.scale, 1. + options.scale)
            elif isinstance(options.scale, bool) and options.scale is True:
                kwargs['scale'] = (0.85, 1.15)

            transforms_list.append(transforms.RandomAffine(**kwargs))

        if isinstance(options.normalize, tuple):
            transforms_list.append(transforms.Normalize(mean=[options.normalize[0]], std=[options.normalize[1]]))

        return transforms_list

    @staticmethod
    def get_augmentations(options: DataAugmentationOptions, include_affine=True):
        return transforms.Compose([tf for tf in DataAugmentationUtils.__get_transforms(options) if
                                   (include_affine or not isinstance(tf, transforms.RandomAffine))])

    @staticmethod
    def get_scriptable_augmentation(options: DataAugmentationOptions):
        all_transforms = [tf for tf in DataAugmentationUtils.__get_transforms(options) if
                          not isinstance(tf, (transforms.RandomAffine, transforms.ToTensor))]
        return nn.Sequential(
            *all_transforms
        )

    @staticmethod
    def determine_mean_std_for_augmentation(options: DataAugmentationOptions, train_set_path: str):
        all_transforms = DataAugmentationUtils.__get_transforms(options)
        img_folder = ImageFolder(train_set_path, transform=transforms.Compose(
            [tf for tf in all_transforms if not isinstance(tf, transforms.RandomAffine)]))
        dataloader_std_mean = DataLoader(img_folder, batch_size=8, shuffle=True, num_workers=8)

        mean_sum = 0.
        n_total = 0
        var_sum = 0.

        for imgs, _ in dataloader_std_mean:
            for img in imgs:
                mean_sum += img.mean().item()
                n_total += 1
        mean = mean_sum / n_total

        for imgs, _ in dataloader_std_mean:
            for img in imgs:
                stds = ((img - mean) ** 2)
                var = stds.sum().item() / (32 * 32)
                var_sum += var

        std = math.sqrt(var_sum / n_total)
        print(f'Dataset mean: {mean:.4f}, std: {std:.4f}')
        return mean, std
