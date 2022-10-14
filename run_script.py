from argparse import ArgumentParser

from running import RunnerKinderlabor

if __name__ == "__main__":
    parser = ArgumentParser(description='Run Script for Thesis Experiments')
    parser.add_argument('--create-dataset-folders', default=False, action='store_true',
                        help='Create Dataset Folders for experiments')
    parser.add_argument('--plot-examples', default=False, action='store_true',
                        help='Plot examples from the Tasks in the Dataset')
    parser.add_argument('--compare-model-sizes', default=False, action='store_true',
                        help='Compare Model Sizes')
    parser.add_argument('--compare-data-collection', default=False, action='store_true',
                        help='Compare Data Collection Methodologies. WARNING: REQUIRES PSEUDONYMOUS DATASET!')
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
    parsed_args = parser.parse_args()

    if parsed_args.create_dataset_folders:
        RunnerKinderlabor.create_dataset_folders()
    if parsed_args.plot_examples:
        RunnerKinderlabor.plot_examples()
    if parsed_args.compare_model_sizes:
        RunnerKinderlabor.compare_model_sizes()
    if parsed_args.compare_data_collection:
        RunnerKinderlabor.compare_data_collection()
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

