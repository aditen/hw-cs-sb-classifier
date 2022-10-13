from argparse import ArgumentParser

from running import RunnerKinderlabor

if __name__ == "__main__":
    parser = ArgumentParser(description='Run Script for Thesis Experiments')
    parser.add_argument('--no-examples', default=False, action='store_true',
                        help='Visualize Examples, omit if you do not want to visualize')
    parsed_args = parser.parse_args()
    if not parsed_args.no_examples:
        RunnerKinderlabor.plot_samples()
