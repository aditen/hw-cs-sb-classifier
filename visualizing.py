import math
import os.path
import os.path
import warnings
from typing import List, Tuple

import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from tqdm import tqdm

from data_loading import DataloaderKinderlabor
from grayscale_model import ModelVersion
from training import TrainerKinderlabor
from utils import class_name_dict, UtilsKinderlabor
from utils import short_names_tasks, TaskType, long_names_tasks, \
    long_names_models


def show_on_axis(ax, img_np, class_name, mean, std, class_name_predicted=None):
    inp = img_np.transpose((1, 2, 0))
    if not math.isnan(mean) and not math.isnan(std):
        mean = np.array([mean])
        std = np.array([std])
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = inp.squeeze()
    ax.imshow(inp, cmap='gray_r', vmin=0, vmax=1)
    title = class_name
    if class_name_predicted is not None:
        title = f'{class_name_predicted} ({class_name})'
    ax.set_title(title)


class VisualizerKinderlabor:
    def __init__(self, data_loader: DataloaderKinderlabor, save_plots_to_disk=True, run_id=None):
        self.__run_id = run_id if run_id is not None else data_loader.get_folder_name()
        self.__visualization_dir = f'output_visualizations/{self.__run_id}'
        if not os.path.isdir(self.__visualization_dir):
            os.mkdir(self.__visualization_dir)
        self.__data_loader = data_loader
        self.__save_plots_to_disk = save_plots_to_disk

    def visualize_some_train_samples(self):
        train_loader, valid_loader, test_loader = self.__data_loader.get_data_loaders()
        mean, std = self.__data_loader.get_mean_std()
        # Get a batch of training data
        inputs, classes = next(iter(train_loader))

        fig, axs = plt.subplots(ncols=4, nrows=math.ceil(len(classes) / 4))
        fig.set_figheight(6.4 * len(classes) / 16)

        class_names = [class_name_dict[c] for c in self.__data_loader.get_classes()]
        class_names.append("unknown")

        for img_idx in range(len(inputs)):
            show_on_axis(axs.flat[img_idx], inputs[img_idx, :, :].cpu().numpy(),
                         class_names[classes[img_idx]], mean, std)
        fig.tight_layout()
        if self.__save_plots_to_disk:
            plt.savefig(f'{self.__visualization_dir}/train_samples.pdf')
        plt.show()

    def visualize_training_progress(self, trainer: TrainerKinderlabor):
        epochs, train_loss, valid_loss, train_acc, valid_acc = trainer.get_training_progress()
        if len(epochs) > 0:
            fig, ax = plt.subplots()
            ax.plot(epochs, train_loss, label="Train Loss")
            ax.plot(epochs, valid_loss, label="Validation Loss")
            ax.plot(epochs, train_acc, label="Training Accuracy")
            ax.plot(epochs, valid_acc, label="Validation Accuracy")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel("Epoch")
            plt.ylabel("Model Performance")
            plt.title("Model Performance over Epochs")
            plt.legend()

            if self.__save_plots_to_disk:
                plt.savefig(
                    f'{self.__visualization_dir}/learning_progress.pdf')
            plt.show()
        else:
            print("No training done yet! Please call this function after training")

    def visualize_model_errors(self, trainer: TrainerKinderlabor):
        # reminder: err_samples is tuple of form (img, actual, pred)
        actual, predicted, best_probs, err_samples, _, loader, __ = trainer.get_predictions()
        if loader != self.__data_loader:
            print("Loaders are different! Please check you provide the right instance to the visualizer!")
            return
        mean, std = self.__data_loader.get_mean_std()
        actual_kn, predicted_kn = zip(*((ac, pr) for ac, pr in zip(actual, predicted) if ac != -1))

        if -1 in [err[1] for err in err_samples]:
            imgs_uu, actual_uu, pred_uu, probs_uu = zip(
                *((err, ac, pr, prob) for err, ac, pr, prob in err_samples if ac == -1))
            num_errs_to_show = min(64, len(imgs_uu))
            fig, axs = plt.subplots(nrows=math.ceil(num_errs_to_show / 4), ncols=4)
            fig.set_figheight(6.4 * num_errs_to_show / 16)
            for img_idx in range(num_errs_to_show):
                show_on_axis(ax=axs.flat[img_idx], img_np=imgs_uu[img_idx],
                             class_name=f'{probs_uu[img_idx] * 100:.2f}%',
                             mean=mean, std=std, class_name_predicted=class_name_dict[
                        self.__data_loader.get_classes()[pred_uu[img_idx]]])
            fig.tight_layout()
            if self.__save_plots_to_disk:
                plt.savefig(
                    f'{self.__visualization_dir}/test_errors_unknown.pdf')
            plt.show()

        if len([1 for x in err_samples if x[1] != -1]) > 0:
            imgs_err_kn, actual_err_kn, pred_err_kn = zip(
                *((err, ac, pr) for err, ac, pr, prob in err_samples if ac != -1))
            num_errs_to_show = min(64, len(imgs_err_kn))
            print(f'Num errors shown in errors plot (max 64): {num_errs_to_show}/{len(imgs_err_kn)}')
            fig, axs = plt.subplots(nrows=math.ceil(num_errs_to_show / 4), ncols=4)
            fig.set_figheight(6.4 * num_errs_to_show / 16)
            for img_idx in range(num_errs_to_show):
                show_on_axis(axs.flat[img_idx], imgs_err_kn[img_idx],
                             class_name_dict[self.__data_loader.get_classes()[actual_err_kn[img_idx]]],
                             mean, std, class_name_predicted=class_name_dict[
                        self.__data_loader.get_classes()[pred_err_kn[img_idx]]])
            fig.tight_layout()
            if self.__save_plots_to_disk:
                plt.savefig(
                    f'{self.__visualization_dir}/test_errors_known.pdf')
            plt.show()
        else:
            print("Theoretically perfect model for knowns, no error on known classes to plot!")

        ConfusionMatrixDisplay.from_predictions(actual_kn, predicted_kn,
                                                display_labels=[class_name_dict[x] for x in loader.get_classes()],
                                                im_kw={"norm": matplotlib.colors.SymLogNorm(linthresh=1)})
        if self.__save_plots_to_disk:
            plt.savefig(
                f'{self.__visualization_dir}/conf_matrix.pdf')
        plt.show()

    def visualize_class_distributions(self):
        train_df, valid_df, test_df, all_df = self.__data_loader.get_set_dfs()
        labels = all_df['label'].unique()
        vals_train = [len(train_df[train_df['label'] == label]) for label in labels]
        vals_valid = [len(valid_df[valid_df['label'] == label]) for label in labels]
        vals_test = [len(test_df[test_df['label'] == label]) for label in labels]

        labels = [class_name_dict[x] for x in labels]

        fig, ax = plt.subplots()

        ax.bar(labels, vals_train, label='Train Set')
        ax.bar(labels, vals_valid, label='Validation Set', bottom=vals_train)
        ax.bar(labels, vals_test, label='Test Set', bottom=[sum(x) for x in zip(vals_train, vals_valid)])

        ax.set_ylabel('Number of Samples')
        ax.set_title('Number of Samples per Class and Set')
        ax.legend()

        if self.__save_plots_to_disk:
            plt.savefig(
                f'{self.__visualization_dir}/class_dist.pdf')
        plt.show()

    def visualize_2d_space(self, trainer: TrainerKinderlabor):
        UtilsKinderlabor.random_seed()
        actual, predicted, best_probs, _, coords, loader, __ = trainer.get_predictions()
        if loader != self.__data_loader:
            print("Loaders are different! Please check you provide the right instance to the visualizer!")
            return
        fig, ax = plt.subplots()
        xes = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        labels = [(class_name_dict[loader.get_classes()[x]] if x >= 0 else "unknown") for x in actual]
        colors = ['red', 'green', 'blue', 'orange', 'yellow', 'pink', 'darkred', 'gold', 'cyan', 'olive',
                  'brown', 'purple', 'lime']
        already_plotted_legends = set()
        vis_indices = np.random.choice(len(labels), min(len(labels), 1000))
        for i in tqdm(vis_indices, unit="Coordinates"):
            # Plot unknowns as black color
            color = colors[actual[i]] if actual[i] >= 0 else "black"
            if labels[i] not in already_plotted_legends:
                ax.scatter(xes[i], ys[i], label=labels[i], color=color, alpha=0.25)
                already_plotted_legends.add(labels[i])
            else:
                ax.scatter(xes[i], ys[i], color=color, alpha=0.25)

        plt.legend()
        if self.__save_plots_to_disk:
            plt.savefig(
                f'{self.__visualization_dir}/scatter.pdf')
        plt.show()

    def visualize_prob_histogram(self, trainer: TrainerKinderlabor):
        actual, predicted, best_probs, _, coords, loader, __ = trainer.get_predictions()
        if loader != self.__data_loader:
            print("Loaders are different! Please check you provide the right instance to the visualizer!")
            return

        probs_known = [prob for (prob, act, pred) in zip(best_probs, actual, predicted) if (act != -1 and act == pred)]
        probs_unknown = [prob for (prob, act) in zip(best_probs, actual) if act == -1]

        plt.hist(probs_known, bins=50, label="Known", histtype='step', color='green')
        if len(probs_unknown) > 0:
            plt.hist(probs_unknown, bins=50, label="Unknown", histtype='step', color='red')

        plt.yscale('log')
        plt.legend()
        if self.__save_plots_to_disk:
            plt.savefig(
                f'{self.__visualization_dir}/probs.pdf')
        plt.show()

    def visualize_open_set_recognition_curve(self, trainers: List[Tuple[str, TrainerKinderlabor]], balanced=True):
        for label, trainer in trainers:
            actual, predicted, best_probs, _, coords, loader, __ = trainer.get_predictions()
            thresh_vals = np.arange(0, 1, 1. / 1000).tolist()
            fps, ccrs, balanced_accs = [], [], []

            for thresh in tqdm(thresh_vals, unit="thresh"):
                prob_vals_uk = [prob for (label, prob) in zip(actual, best_probs) if label < 0]
                n_fp = len([prob for prob in prob_vals_uk if prob > thresh])
                fps.append(n_fp / len(prob_vals_uk))

                prob_vals_k = [prob for (label, prob) in zip(actual, best_probs) if label >= 0]
                labels_k = [label for (label, prob) in zip(actual, best_probs) if label >= 0]
                pred_k = [pred for (label, pred) in zip(actual, predicted) if label >= 0]
                n_ccr = [1 for (label, pred, prob) in zip(labels_k, pred_k, prob_vals_k) if
                         (label == pred and prob >= thresh)]
                ccrs.append(len(n_ccr) / len(prob_vals_k))

                for i in range(len(pred_k)):
                    if prob_vals_k[i] < thresh:
                        pred_k[i] = -1
                warnings.filterwarnings('ignore', category=UserWarning)
                balanced_acc = balanced_accuracy_score(labels_k, pred_k)
                balanced_accs.append(balanced_acc)

            if balanced:
                plt.plot(fps, balanced_accs, label=label)
            else:
                plt.plot(fps, ccrs, label=label)
        plt.xlabel("False Positive Rate")
        plt.ylabel("Balanced Accuracy" if balanced else "Correct Classification Rate")
        # plt.title(("Balanced " if balanced else "") + "Open Set Recognition Curve")
        plt.xscale('log')
        plt.legend()
        if self.__save_plots_to_disk:
            plt.savefig(
                f'{self.__visualization_dir}/osrc.pdf')
        plt.show()

    @staticmethod
    def visualize_baseline_results_as_plot(csv_path: str):
        # see https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        pd.options.mode.chained_assignment = None
        full_df = pd.read_csv(csv_path, sep=";")
        sns.set_theme(font_scale=1.75, style='whitegrid')
        fig = plt.figure(figsize=(21., 19.2))
        all_tasks = list(TaskType)
        all_models = [ModelVersion.SM, ModelVersion.LE_NET]
        grid = plt.GridSpec(len(all_tasks), 1)

        for task_type in all_tasks:
            fake = fig.add_subplot(grid[all_tasks.index(task_type)])
            # '\n' remains important
            fake.set_title(f'{long_names_tasks[task_type]}\n', fontweight='semibold', size=28)
            fake.set_axis_off()

            gs = GridSpecFromSubplotSpec(1, len(all_models),
                                         subplot_spec=grid[all_tasks.index(task_type)])

            model_idx = 0
            for i, model in enumerate(all_models):
                ax = fig.add_subplot(gs[model_idx])
                df_loop = full_df[(full_df['model'] == model.name) & (full_df['task'] == short_names_tasks[task_type])]
                df_loop.performance *= 100
                df_loop.performance = 100 - df_loop.performance
                sns.barplot(
                    data=df_loop, errorbar=None,
                    x="split", y="performance", hue="augmentation",
                    alpha=.6, ax=ax
                )
                # not needed because there is enough visual space
                for j, (tick) in enumerate(ax.xaxis.get_major_ticks()):
                    if (j % 2) != (i % 2):
                        pass
                        # tick.set_visible(False)
                ax.set_title(f'{long_names_models[model]}', size=25)
                ax.get_legend().remove()

                ax.set_xlabel('Data Split')
                if model_idx == 0:
                    ax.set_ylabel('Error Rate (%)')
                else:
                    ax.set_ylabel('Error Rate (%)')
                #    ax.set_ylabel(None)
                model_idx += 1

        # add legend
        handles, labels = fig.axes[2].get_legend_handles_labels()
        fig.axes[2].legend(handles, labels, bbox_to_anchor=(1, 1.03), title="Augmentation")
        fig.tight_layout()
        plt.show()
        fig.savefig("./output_visualizations/base_plot.pdf", dpi=75)

    @staticmethod
    def plot_samples(indices, n_cols, n_rows):
        ims = []
        for idx in indices:
            ims.append(Image.open(f'{DataloaderKinderlabor.IMG_CSV_FOLDER}{idx}.jpeg'))

        fig = plt.figure(figsize=(n_cols, n_rows))
        grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1, share_all=True)

        for ax, im in zip(grid, ims):
            im = im.resize((32, 32))
            ax.imshow(im)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        fig.tight_layout()
        return fig

    @staticmethod
    def visualize_methodology_comparison(sheet_df: pd.DataFrame, single_booklet_df: pd.DataFrame,
                                         mini_booklet_df: pd.DataFrame, booklet_df: pd.DataFrame):
        if not os.path.isdir('output_visualizations/data_availability'):
            os.mkdir('output_visualizations/data_availability')
        for task_type in list(TaskType):
            task_type = task_type.value
            labels = sheet_df[sheet_df['type'] == task_type]['label'].unique()
            vals_booklet = [len(booklet_df[(booklet_df['label'] == label) & (booklet_df['type'] == task_type)]) for
                            label in
                            labels]
            vals_single_booklet = [len(single_booklet_df[(single_booklet_df['label'] == label) & (
                    single_booklet_df['type'] == task_type)]) for label in labels]
            vals_sheet = [len(sheet_df[(sheet_df['label'] == label) & (sheet_df['type'] == task_type)]) for label in
                          labels]
            vals_mini_booklet = [
                len(mini_booklet_df[(mini_booklet_df['label'] == label) & (mini_booklet_df['type'] == task_type)]) for
                label
                in labels]

            labels = [class_name_dict[x] for x in labels]
            fig, ax = plt.subplots()

            x = np.arange(len(labels))  # the label locations
            width = 0.2  # the width of the bars

            ax.bar(x - 1.5 * width, vals_sheet, width, label='Sheet')
            ax.bar(x - 0.5 * width, vals_mini_booklet, width, label='Mini-Booklet')
            ax.bar(x + 0.5 * width, vals_booklet, width, label='Booklet')
            ax.bar(x + 1.5 * width, vals_single_booklet, width, label='1x Fully Solved Booklet')

            ax.set_ylabel('Number of Samples')
            ax.set_xlabel('Symbol')
            # no title because caption is title in thesis
            # ax.set_title(f'Number of Samples per Methodology for type {task_type}')
            ax.set_xticks(x, labels)
            ax.set_yscale('log')
            ax.legend()

            plt.show()
            fig.savefig(
                f'output_visualizations/data_availability/sample_comp_{task_type}.pdf')
