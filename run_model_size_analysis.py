from tabulate import tabulate

from grayscale_model import ModelVersion, get_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    size_map = [[version.name, count_parameters(get_model(version, num_classes=5))] for version in ModelVersion]
    print(f'Sizes of different models: {size_map}')
    print(tabulate(size_map, headers=["Model", "Num Parameters"]))
