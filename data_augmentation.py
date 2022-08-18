import math

from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


class DataAugmentation:
    def __init__(self, auto_contrast=True, invert=True, normalize=False, rotate=True, translate=True, scale=True):
        # Grayscale and Resize are always present
        transforms_list = [transforms.Grayscale(),
                           transforms.Resize((32, 32))]

        # Auto Contrast is yes or no
        if auto_contrast:
            transforms_list.append(transforms.RandomAutocontrast(p=1.))

        # Inverting is yes or no
        if invert:
            transforms_list.append(transforms.RandomInvert(p=1.))

        # Rotation, translation or scaling can be
        if rotate or translate or scale:
            kwargs = {}
            # Rotation degrees
            if isinstance(rotate, tuple):
                kwargs['degrees'] = rotate
            elif isinstance(rotate, int):
                kwargs['degrees'] = (-rotate, rotate)
            elif isinstance(rotate, bool) and rotate is True:
                kwargs['degrees'] = (-30, 30)

            # Translation values (relative)
            if isinstance(translate, tuple):
                kwargs['translate'] = translate
            elif isinstance(translate, float):
                kwargs['translate'] = (translate, translate)
            elif isinstance(translate, bool) and translate is True:
                kwargs['translate'] = (0.15, 0.15)

            # Scaling values (relative)
            if isinstance(scale, tuple):
                kwargs['scale'] = scale
            elif isinstance(scale, float):
                kwargs['scale'] = (1. - scale, 1. + scale)
            elif isinstance(scale, bool) and scale is True:
                kwargs['scale'] = (0.85, 1.15)

            transforms_list.append(transforms.RandomAffine(**kwargs))

        if isinstance(normalize, tuple):
            transforms_list.append(transforms.Normalize(mean=[normalize[0]], std=[normalize[1]]))

        elif isinstance(normalize, DataLoader):
            mean_sum = 0.
            n_total = 0
            var_sum = 0.

            for imgs, _ in normalize:
                for img in imgs:
                    mean_sum += img.mean().item()
                    n_total += 1
            self.__mean = mean_sum / n_total

            for imgs, _ in normalize:
                for img in imgs:
                    stds = ((img - self.__mean) ** 2)
                    var = stds.sum().item() / (32 * 32)
                    var_sum += var

            self.__std = math.sqrt(var_sum / n_total)
            print(f'Dataset mean: {self.__mean:.4f}, std: {self.__std:.4f}')
            transforms_list.append(transforms.Normalize(mean=[self.__mean], std=[self.__std]))

        self.__transforms_list = transforms_list

    def get_data_augmentations_composed(self, test=False, exclude_normalize=False):
        all_transforms = self.__transforms_list + [transforms.ToTensor()]
        if test:
            all_transforms = [x for x in all_transforms if not isinstance(x, transforms.RandomAffine)]
        if exclude_normalize:
            all_transforms = [x for x in all_transforms if not isinstance(x, transforms.Normalize)]
        return transforms.Compose(all_transforms)

    def get_data_augmentations_as_module_for_prediction(self):
        all_transforms = [x for x in self.__transforms_list if not isinstance(x, transforms.RandomAffine)]
        return nn.Sequential(
            *all_transforms
        )

    def get_mean_std(self):
        return self.__mean, self.__std
