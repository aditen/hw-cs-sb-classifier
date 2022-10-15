import math
import os
import shutil
from typing import Optional
from typing import Tuple, List

import pandas as pd
import torchvision.transforms
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pandas import DataFrame
from tabulate import tabulate

from data_augmentation import DataAugmentationOptions, DataAugmentationUtils
from data_loading import DataloaderKinderlabor
from grayscale_model import ModelVersion, get_model
from training import TrainerKinderlabor
from utils import UtilsKinderlabor, data_split_dict, TaskType, DataSplit
from utils import short_names_tasks, short_names_models, long_names_tasks, \
    Unknowns, short_names_losses, LossFunction
from visualizing import VisualizerKinderlabor

csv_path_baseline = './output_visualizations/runs_base.csv'


def get_default_loss(task_type: TaskType):
    return LossFunction.BCE if task_type == TaskType.CROSS else LossFunction.SOFTMAX


def get_run_id(prefix: str, task_type: TaskType, aug_name: str, data_split: DataSplit, model: ModelVersion,
               loss: LossFunction, training_unknowns: Optional[Unknowns]):
    return f"{prefix}_tsk[{short_names_tasks[task_type]}]_agm[{aug_name}]" \
           f"_spl[{data_split_dict[data_split]}]_mdl[{short_names_models[model]}]_ls[{short_names_losses[loss]}]" \
           f"_tu[{'none' if training_unknowns is None else training_unknowns.name}]"


class RunnerKinderlabor:
    @staticmethod
    def admin_create_dataset():
        def __split_random(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
            train_df = df.sample(frac=0.8)
            max_samples_per_class = 1500
            for label in train_df['label'].unique():
                label_df = train_df.loc[train_df['label'] == label]
                if len(label_df) > max_samples_per_class:
                    keep_df = label_df.sample(n=max_samples_per_class)
                    drop_df = label_df.drop(keep_df.index)
                    train_df = train_df.drop(drop_df.index)
            test_df = df.drop(train_df.index)
            return train_df, test_df

        def __split_hold_out(df: DataFrame, hold_out: List[str]) -> Tuple[DataFrame, DataFrame]:
            test_df = df[df['class'].isin(hold_out)]
            train_df = df.drop(test_df.index)
            return train_df, test_df

        def __split_sheets_booklets(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
            train_df = df[
                (df['sheet'] == 'Datensammelblatt Kinderlabor') |
                (df['sheet'] == 'Data Collection 1. Klasse') |
                ((df['student'] == 'Laura_Heft_komplett_Test') & (df['label'] == 'EMPTY'))
                ]
            test_df = df.drop(train_df.index)
            return train_df, test_df

        UtilsKinderlabor.random_seed()
        full_df = DataloaderKinderlabor.raw_herby_df()
        hold_out_classes = os.getenv('HOLDOUTCLASSES', default=None)
        if hold_out_classes is None:
            raise ValueError('Hold Out classes env variable not defined!')
        hold_out_classes = hold_out_classes.split(",")
        print(f"Hold out classes: {hold_out_classes}")

        kwargs_split_cols = {data_split_dict[val]: "" for val in DataSplit}
        full_df = full_df.assign(**kwargs_split_cols)

        # INSPECT does not belong to any splits!
        df = full_df[full_df['label'] != 'INSPECT']

        for split in DataSplit:
            for task_type in TaskType:
                task_df = df[df['type'] == task_type.value]
                if split == DataSplit.RANDOM:
                    train, test = __split_random(task_df)
                elif split == DataSplit.HOLD_OUT_CLASSES:
                    train, test = __split_hold_out(task_df, hold_out=hold_out_classes)
                elif split == DataSplit.TRAIN_SHEETS_TEST_BOOKLETS:
                    train, test = __split_sheets_booklets(task_df)
                else:
                    raise ValueError(f'No function for split {split.name}')
                valid = train.sample(frac=0.15)
                train = train.drop(valid.index)
                full_df.loc[train.index, data_split_dict[split]] = "train"
                full_df.loc[valid.index, data_split_dict[split]] = "valid"
                full_df.loc[test.index, data_split_dict[split]] = "test"

        full_df = full_df.drop(['request', 'student', 'class', 'sheet', 'field', 'exercise'], axis=1)
        full_df.to_csv('./kinderlabor_dataset/dataset_anonymized.csv', sep=";", index_label="id")

        for idx, row in full_df.iterrows():
            shutil.copy(
                f'C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/20220925_corr_v2/{idx}.jpeg',
                f'./kinderlabor_dataset/{idx}.jpeg')

    @staticmethod
    def create_dataset_folders():
        for data_split in DataSplit:
            # avoid double work forced
            if data_split == DataSplit.HOLD_OUT_CLASSES:
                continue
            for task_type in TaskType:
                loader = DataloaderKinderlabor(task_type=task_type, data_split=data_split, force_reload_data=True)
                visualizer = VisualizerKinderlabor(loader,
                                                   run_id=f"data_split[{data_split_dict[data_split]}]"
                                                          f"_task[{long_names_tasks[task_type]}]")
                visualizer.visualize_class_distributions()
                visualizer.visualize_some_train_samples()
        # create unknown folders
        for task_type in TaskType:
            DataloaderKinderlabor(task_type=task_type, force_reload_data=True,
                                  data_split=DataSplit.HOLD_OUT_CLASSES,
                                  known_unknowns=Unknowns.ALL_OF_TYPE)
            loader = DataloaderKinderlabor(task_type=task_type, force_reload_data=True,
                                           data_split=DataSplit.HOLD_OUT_CLASSES,
                                           known_unknowns=Unknowns.HOLD_OUT_CLASSES,
                                           unknown_unknowns=Unknowns.HOLD_OUT_CLASSES)
            visualizer = VisualizerKinderlabor(loader,
                                               run_id=f"data_split[{data_split_dict[DataSplit.HOLD_OUT_CLASSES]}]"
                                                      f"_task[{long_names_tasks[task_type]}]")
            visualizer.visualize_class_distributions()
            visualizer.visualize_some_train_samples()
        # load other unknown datasets if not yet on disk - no force as kinderlabor ones are already created!
        for uk_type in Unknowns:
            if uk_type == Unknowns.HOLD_OUT_CLASSES or uk_type == Unknowns.HOLD_OUT_CLASSES:
                continue
            DataloaderKinderlabor(task_type=TaskType.COMMAND, force_reload_data=False,
                                  data_split=DataSplit.HOLD_OUT_CLASSES,
                                  known_unknowns=uk_type,
                                  unknown_unknowns=uk_type)

    @staticmethod
    def plot_examples():
        EMPTIES = [98, 312, 3320, 35317, 35883]
        PLUS_ONES = [32489, 33289, 3988, 31382, 46818]
        MINUS_ONES = [96, 837, 3241, 33279, 33288]
        TURN_RIGHTS = [914, 1523, 33321, 34428, 44975]
        TURN_LEFTS = [1089, 31733, 33672, 47671, 47526]
        UK_BASE = [24116, 30640, 40011, 41476, 45671]

        LOOP_TWICES = [683, 869, 43024, 44744, 45594]
        LOOP_THREES = [841, 3246, 43035, 43741, 44253]
        LOOP_FOURS = [842, 3514, 42862, 44003, 47927]
        LOOP_ENDS = [899, 1299, 2785, 43366, 44228]
        UK_LOOP = [765, 3494, 7473, 30062, 39421]

        EMPTIES_ORT = [657, 30251, 30622, 35213, 36340]
        ARROWS_DOWN = [944, 2987, 8321, 44503, 45557]
        ARROWS_RIGHT = [1641, 46901, 43474, 3180, 2305]
        ARROWS_UP = [1467, 1779, 3906, 43455, 47915]
        ARROWS_LEFT = [1219, 3069, 31659, 43630, 44844]
        UK_ARR = [23638, 35705, 37222, 50685, 54700]

        EMPTIES_CRS = [193, 1440, 1730, 47579, 35789]
        CROSSES = [947, 1627, 2102, 2446, 33667]
        UK_CRS = [8051, 15809, 26663, 34661, 36270]

        BASIC_INSTRUCTION_INDICES = EMPTIES + PLUS_ONES + MINUS_ONES + TURN_RIGHTS + TURN_LEFTS + UK_BASE
        ADVANCED_INSTRUCTION_INDICES = LOOP_TWICES + LOOP_THREES + LOOP_FOURS + LOOP_ENDS + UK_LOOP

        ORIENTATION_INDICES = EMPTIES_ORT + ARROWS_DOWN + ARROWS_RIGHT + ARROWS_UP + ARROWS_LEFT + UK_ARR
        CROSS_INDICES = EMPTIES_CRS + CROSSES + UK_CRS

        fig1 = VisualizerKinderlabor.plot_samples(BASIC_INSTRUCTION_INDICES, 5, 6)
        fig1.savefig('./output_visualizations/observations_in_basic_instructions.pdf')
        plt.show()

        fig2 = VisualizerKinderlabor.plot_samples(ADVANCED_INSTRUCTION_INDICES, 5, 5)
        fig2.savefig('./output_visualizations/observations_in_advanced_instructions.pdf')
        plt.show()

        fig3 = VisualizerKinderlabor.plot_samples(ORIENTATION_INDICES, 5, 6)
        fig3.savefig('./output_visualizations/observations_in_orientation.pdf')
        plt.show()

        fig4 = VisualizerKinderlabor.plot_samples(CROSS_INDICES, 5, 3)
        fig4.savefig('./output_visualizations/observations_in_crosses.pdf')
        plt.show()

    @staticmethod
    def compare_model_sizes():
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        size_map = [[version.name, count_parameters(get_model(version, num_classes=5))] for version in ModelVersion]
        print(tabulate(size_map, headers=["Model", "# of learnable Parameters", "# of Layers"]))

    @staticmethod
    def plot_augmentations():
        vis_augs = [DataAugmentationOptions(to_tensor=False, grayscale=False, invert=False),
                    DataAugmentationOptions(to_tensor=False, invert=False),
                    DataAugmentationOptions(to_tensor=False),
                    DataAugmentationOptions(to_tensor=True, gaussian_noise_sigma=0.05),
                    DataAugmentationOptions(to_tensor=True, gaussian_noise_sigma=0.15),
                    DataAugmentationOptions(to_tensor=True, gaussian_noise_sigma=0.15, auto_contrast=True),
                    DataAugmentationOptions(to_tensor=False, auto_contrast=True),
                    DataAugmentationOptions(to_tensor=False, equalize=True),
                    DataAugmentationOptions(to_tensor=False, rotate=(-90, 90)),
                    DataAugmentationOptions(to_tensor=False, translate=(0.5, 0.5)),
                    DataAugmentationOptions(to_tensor=False, scale=(0.5, 1.5)),
                    DataAugmentationOptions(to_tensor=False, crop_center=True)]
        titles = ["Original", "Grayscale", "Invert", "Noise 1", "Noise 2", "Noise 2+", "Contrast", "Equalize", "Rotate",
                  "Translate", "Scale", "Crop"]

        all_ids_to_show = [312, 1089, 31382, 34428, 43024, 1299]

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

    @staticmethod
    def admin_compare_data_collection():
        if not os.path.isfile(
                'C:/Users/41789/Documents/uni/ma/kinderlabor_unterlagen/train_data/20220925_corr_v2/dataset.csv'):
            raise ValueError('Did not find non-anonymized dataset on your machine! Please contact the admins')
        df = DataloaderKinderlabor.raw_herby_df()
        classes = df['class'].unique()
        print(f'classes currently in df: {classes}')
        sheets = df['sheet'].unique()
        print(f'Sheets in df: {sheets}')

        task_types = df['type'].unique().tolist()
        print(f'task types: {task_types}')

        sheet_df = df[(df['sheet'] == 'Datensammelblatt Kinderlabor')
                      | (df['sheet'] == 'Data Collection 1. Klasse')]
        single_booklet_df = df[(df['student'] == 'Laura_Heft_komplett_Test')]
        mini_booklet_df = df[(df['sheet'].str.startswith('Mini-Booklet'))]
        booklet_df = df[df['sheet'].str.startswith("Kinderlabor")]
        VisualizerKinderlabor.visualize_methodology_comparison(sheet_df, single_booklet_df, mini_booklet_df, booklet_df)

    @staticmethod
    def train_baseline():
        run_configs = [
            ("none", DataAugmentationOptions.none_aug()),
            ("ac", DataAugmentationOptions.auto_ctr_aug()),
            ("eq", DataAugmentationOptions.eq_aug()),
            ("geo", DataAugmentationOptions.geo_aug()),
            ("geo_ac", DataAugmentationOptions.geo_ac_aug()),
            ("crop", DataAugmentationOptions.crop_aug()),
            ("crop_plus", DataAugmentationOptions.crop_plus_aug()),
        ]
        data_arr = []
        for task_type in TaskType:
            for run_config in run_configs:
                for data_split in DataSplit:
                    for model in [ModelVersion.SM, ModelVersion.LE_NET]:
                        run_id = get_run_id(prefix="base", task_type=task_type, aug_name=run_config[0],
                                            data_split=data_split, model=model, loss=get_default_loss(task_type),
                                            training_unknowns=None)
                        print(
                            f'Running for augmentation {run_config[0]} and data split {data_split.value} '
                            f'and task {task_type.name} using model {model.name}')

                        # Initialize data loader: data splits and loading from images from disk
                        loader = DataloaderKinderlabor(task_type=task_type,
                                                       data_split=data_split,
                                                       augmentation_options=run_config[1])

                        # visualize class distribution and some (train) samples
                        visualizer = VisualizerKinderlabor(loader, run_id=run_id)

                        # Train model and analyze training progress (mainly when it starts overfitting on validation set)
                        trainer = TrainerKinderlabor(loader, run_id,
                                                     load_model_from_disk=True,
                                                     loss_function=get_default_loss(task_type),
                                                     model_version=model)
                        trainer.train_model(n_epochs=75, lr=0.01, sched=(10, 0.5), n_epochs_wait_early_stop=10)
                        visualizer.visualize_training_progress(trainer)

                        # Predict on test samples
                        trainer.predict_on_test_samples()
                        visualizer.visualize_model_errors(trainer)
                        trainer.script_model()

                        data_arr.append(
                            [run_config[0], data_split_dict[data_split], short_names_tasks[task_type],
                             trainer.get_predictions()[6], model.name])
        print(tabulate([[x[1], x[2], x[4], x[0], f'{(x[3] * 100):.2f}%'] for x in data_arr],
                       headers=['Data Split', 'Task', 'Model', 'Augmentation', 'Performance']))
        df = pd.DataFrame.from_dict({'augmentation': [row[0] for row in data_arr],
                                     'split': [row[1] for row in data_arr],
                                     'task': [row[2] for row in data_arr],
                                     'performance': [row[3] for row in data_arr],
                                     'model': [row[4] for row in data_arr]})
        df.to_csv('./output_visualizations/runs_base.csv', sep=';')

    @staticmethod
    def plot_baseline():
        if os.path.isfile(csv_path_baseline):
            VisualizerKinderlabor.visualize_baseline_results_as_plot(csv_path_baseline)
        else:
            raise ValueError('CSV of baseline results not found! Please train baseline models first.')

    @staticmethod
    def evaluate_unknowns_on_closed_set():
        plot_tuples_all = []
        loader = None
        for task_type in TaskType:
            run_id = get_run_id(prefix="base", task_type=task_type, aug_name="geo_ac",
                                data_split=DataSplit.HOLD_OUT_CLASSES, model=ModelVersion.SM,
                                loss=get_default_loss(task_type), training_unknowns=None)

            # Initialize data loader: data splits and loading from images from disk
            loader = DataloaderKinderlabor(task_type=task_type,
                                           data_split=DataSplit.HOLD_OUT_CLASSES,
                                           unknown_unknowns=Unknowns.ALL_OF_TYPE,
                                           augmentation_options=DataAugmentationOptions.geo_ac_aug())

            # visualize class distribution and some (train) samples
            visualizer = VisualizerKinderlabor(loader, run_id=f"{run_id}_open_set")

            # Train model and analyze training progress (mainly when it starts overfitting on validation set)
            trainer = TrainerKinderlabor(loader, run_id,
                                         load_model_from_disk=True,
                                         loss_function=get_default_loss(task_type),
                                         model_version=ModelVersion.SM)
            trainer.train_model(n_epochs=75)
            visualizer.visualize_training_progress(trainer)
            trainer.predict_on_test_samples()
            visualizer.visualize_prob_histogram(trainer)
            visualizer.visualize_model_errors(trainer)
            visualizer.visualize_2d_space(trainer)
            plot_tuples_all.append((long_names_tasks[task_type], trainer))
        visualizer = VisualizerKinderlabor(loader, run_id="closed_set_on_unknown")
        visualizer.visualize_open_set_recognition_curve(plot_tuples_all)

    @staticmethod
    def compare_training_unknowns():
        task_type = TaskType.COMMAND
        all_trainers = []
        for uk_type in [None, Unknowns.FAKE_DATA, Unknowns.MNIST, Unknowns.EMNIST_LETTERS, Unknowns.GAUSSIAN_NOISE_015,
                        Unknowns.GAUSSIAN_NOISE_005]:
            loss_fc = LossFunction.ENTROPIC if uk_type is not None else get_default_loss(task_type)
            run_id = get_run_id(prefix="base" if uk_type is None else "os", task_type=task_type, aug_name="geo_ac",
                                data_split=DataSplit.HOLD_OUT_CLASSES, model=ModelVersion.SM,
                                loss=get_default_loss(task_type), training_unknowns=uk_type)

            # Initialize data loader: data splits and loading from images from disk
            loader = DataloaderKinderlabor(task_type=task_type,
                                           data_split=DataSplit.HOLD_OUT_CLASSES,
                                           known_unknowns=uk_type,
                                           unknown_unknowns=Unknowns.ALL_OF_TYPE,
                                           augmentation_options=DataAugmentationOptions.geo_ac_aug())

            # visualize class distribution and some (train) samples
            visualizer = VisualizerKinderlabor(loader, run_id=run_id)
            if uk_type is not None:
                visualizer.visualize_some_train_samples()

            print(
                f'Running for uknowns {"none" if uk_type is None else uk_type.name}')

            # Train model and analyze training progress (mainly when it starts overfitting on validation set)
            trainer = TrainerKinderlabor(loader, run_id,
                                         load_model_from_disk=True,
                                         model_version=ModelVersion.SM,
                                         loss_function=loss_fc)
            trainer.train_model(n_epochs=125, sched=(50, 0.5), n_epochs_wait_early_stop=25)
            visualizer.visualize_training_progress(trainer)
            trainer.predict_on_test_samples()
            visualizer.visualize_prob_histogram(trainer)
            visualizer.visualize_model_errors(trainer)
            all_trainers.append((uk_type.value if uk_type is not None else "Softmax", trainer))
            visualizer.visualize_2d_space(trainer)
        visualizer = VisualizerKinderlabor(loader, run_id="os_approaches_osrc")
        visualizer.visualize_open_set_recognition_curve(all_trainers, plot_suffix="_training_unknowns",
                                                        plot_xlim=[0.05, 1])
