import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data_loading import DataloaderKinderlabor

EMPTIES = [98, 312, 3320, 33746, 35883]
PLUS_ONES = [32489, 33289, 3988, 31382, 46818]
MINUS_ONES = [96, 837, 3241, 33279, 33288]
TURN_RIGHTS = [914, 1523, 33321, 34428, 44975]
TURN_LEFTS = [1089, 31733, 33672, 47671, 47526]

LOOP_TWICES = [683, 869, 43024, 44744, 45594]
LOOP_THREES = [841, 3246, 43035, 43741, 44253]
LOOP_FOURS = [842, 3514, 42862, 44003, 47927]
LOOP_ENDS = [899, 1299, 2785, 43366, 44228]

BASIC_INSTRUCTION_INDICES = EMPTIES + PLUS_ONES + MINUS_ONES + TURN_RIGHTS + TURN_LEFTS
ADVANCED_INSTRUCTION_INDICES = LOOP_TWICES + LOOP_THREES + LOOP_FOURS + LOOP_ENDS


def plot_samples(full_df, indices, n_cols, n_rows):
    filtered_df = pd.DataFrame(columns=full_df.columns)
    for idx in indices:
        filtered_df = pd.concat([filtered_df, df[idx:(idx + 1)]])
    ims = []
    for i, _ in filtered_df.iterrows():
        ims.append(Image.open(f'{DataloaderKinderlabor.IMG_CSV_FOLDER}{i}.jpeg'))

    fig = plt.figure(figsize=(n_cols, n_rows))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1, share_all=True)

    for ax, im in zip(grid, ims):
        im = im.resize((32, 32))
        ax.imshow(im)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    df = DataloaderKinderlabor.raw_df()
    fig1 = plot_samples(df, BASIC_INSTRUCTION_INDICES, 5, 5)
    fig1.savefig('./output_visualizations/observations_in_basic_instructions.pdf')
    plt.show()

    fig2 = plot_samples(df, ADVANCED_INSTRUCTION_INDICES, 5, 4)
    fig2.savefig('./output_visualizations/observations_in_advanced_instructions.pdf')
