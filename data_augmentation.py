from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


# see https://github.com/pytorch/vision/issues/6192
class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, img):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)

        out = img + self.sigma * torch.randn_like(img)
        out = torch.clamp(out, min=0., max=1.)

        if out.dtype != dtype:
            out = out.to(dtype)

        return out


@dataclass
class DataAugmentationOptions:
    to_tensor: bool = True
    grayscale: bool = True
    auto_contrast: bool = False
    invert: bool = True
    equalize: bool = False
    normalize: Tuple[float, float] | bool = False
    rotate: int | Tuple[int, int] | bool = False
    translate: float | Tuple[float, float] | bool = False
    scale: float | Tuple[float, float] | bool = False
    crop_center: bool = False
    gaussian_noise_sigma: float = None

    @staticmethod
    def none_aug():
        return DataAugmentationOptions()

    @staticmethod
    def auto_ctr_aug():
        return DataAugmentationOptions(auto_contrast=True)

    @staticmethod
    def eq_aug():
        return DataAugmentationOptions(equalize=True)

    @staticmethod
    def geo_aug():
        return DataAugmentationOptions(scale=(0.85, 1.15), translate=(0.2, 0.2))

    @staticmethod
    def geo_ac_aug():
        return DataAugmentationOptions(auto_contrast=True, scale=(0.85, 1.15), translate=(0.2, 0.2))

    @staticmethod
    def crop_aug():
        return DataAugmentationOptions(crop_center=True)

    @staticmethod
    def crop_plus_aug():
        return DataAugmentationOptions(crop_center=True, auto_contrast=True, scale=(0.85, 1.15), translate=(0.2, 0.2),
                                       rotate=(-20, 20))


class DataAugmentationUtils:

    @staticmethod
    def __get_transforms(options: DataAugmentationOptions):
        transforms_list = []

        # because it is not necessary for all the unknown data sets, some are already gray scaled
        if options.grayscale:
            transforms_list.append(transforms.Grayscale())

        # Inverting is yes or no
        if options.invert:
            transforms_list.append(transforms.RandomInvert(p=1.))

        # we always operate on a 32x32 grayscale image
        transforms_list.append(transforms.Resize((32, 32)))

        # Crop center of symbol
        if options.crop_center:
            transforms_list.append(transforms.CenterCrop((22, 22)))
            transforms_list.append(transforms.Resize((32, 32)))

        # Auto Contrast is yes or no
        if options.auto_contrast:
            transforms_list.append(transforms.RandomAutocontrast(p=1.))

        # Equalize is yes or no
        if options.equalize:
            transforms_list.append(transforms.RandomEqualize(p=1.))

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

        # covert to tensor (but not if visualizing pil image)
        if options.to_tensor:
            transforms_list.append(transforms.ToTensor())

        if options.gaussian_noise_sigma is not None and options.gaussian_noise_sigma != 0.:
            transforms_list.append(GaussianNoise(options.gaussian_noise_sigma))

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
