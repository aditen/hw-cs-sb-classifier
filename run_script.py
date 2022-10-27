from argparse import ArgumentParser

import matplotlib.pyplot as plt

from running import RunnerKinderlabor

if __name__ == "__main__":
    parser = ArgumentParser(description='Run Script for Thesis Experiments')
    parser.add_argument('--no-plot-windows', default=False, action='store_true',
                        help='Do not show plot windows interactively. Tip: They are stored on disk, '
                             'check output_visualizations folder')
    parser.add_argument('--admin-create-dataset', default=False, action='store_true',
                        help='ATTENTION: ADMIN only! Create Anonymized dataset')
    parser.add_argument('--create-dataset-folders', default=False, action='store_true',
                        help='Create Dataset Folders for experiments')
    parser.add_argument('--plot-examples', default=False, action='store_true',
                        help='Plot examples from the Tasks in the Dataset')
    parser.add_argument('--compare-model-sizes', default=False, action='store_true',
                        help='Compare Model Sizes')
    parser.add_argument('--admin-compare-data-collection', default=False, action='store_true',
                        help='ATTENTION: ADMIN only! Compare Data Collection Methodologies')
    parser.add_argument('--plot-augmentations', default=False, action='store_true',
                        help='Plot Augmentations on selected Samples from the Dataset')
    parser.add_argument('--train-baseline', default=False, action='store_true',
                        help='Train all baseline models on the Dataset. WARNING: TAKES LONG!')
    parser.add_argument('--plot-baseline', default=False, action='store_true',
                        help='Plot baseline results using matplotlib')
    parser.add_argument('--evaluate-unknowns-on-closed-set', default=False, action='store_true',
                        help='Evaluate unknowns on closet set SoftMax Models')
    parser.add_argument('--compare-training-unknowns', default=False, action='store_true',
                        help='Compare Open Set Performance using different Unknowns in Training')
    parser.add_argument('--compare-unknowns-split', default=False, action='store_true',
                        help='Compare Splitting Unknowns to be in train/validation set as well')
    parsed_args = parser.parse_args()


    def intercept_plot_opener():
        pass


    if parsed_args.no_plot_windows:
        plt.show = intercept_plot_opener
    if parsed_args.admin_create_dataset:
        RunnerKinderlabor.admin_create_dataset()
    if parsed_args.create_dataset_folders:
        RunnerKinderlabor.create_dataset_folders()
    if parsed_args.plot_examples:
        RunnerKinderlabor.plot_examples()
    if parsed_args.compare_model_sizes:
        RunnerKinderlabor.compare_model_sizes()
    if parsed_args.admin_compare_data_collection:
        RunnerKinderlabor.admin_compare_data_collection()
    if parsed_args.plot_augmentations:
        RunnerKinderlabor.plot_augmentations()
    if parsed_args.train_baseline:
        RunnerKinderlabor.train_baseline()
    if parsed_args.plot_baseline:
        RunnerKinderlabor.plot_baseline()
    if parsed_args.evaluate_unknowns_on_closed_set:
        RunnerKinderlabor.evaluate_unknowns_on_closed_set()
    if parsed_args.compare_training_unknowns:
        RunnerKinderlabor.compare_training_unknowns()
    if parsed_args.compare_unknowns_split:
        RunnerKinderlabor.compare_unknowns_split()
