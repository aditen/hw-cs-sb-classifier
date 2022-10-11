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

EMPTIES_ORT = [657, 30251, 30622, 35213, 36340]
ARROWS_DOWN = [944, 2987, 8321, 44503, 45557]
ARROWS_RIGHT = [1641, 46901, 43474, 3180, 2305]
ARROWS_UP = [1467, 1779, 3906, 43455, 47915]
ARROWS_LEFT = [1219, 3069, 31659, 43630, 44844]
# TODO: fill
UK_ARR = []

EMPTIES_CRS = [193, 1440, 1730, 47579, 35789]
CROSSES = [947, 1627, 2102, 2446, 33667]
# TODO: fill
UK_CRS = []

BASIC_INSTRUCTION_INDICES = EMPTIES + PLUS_ONES + MINUS_ONES + TURN_RIGHTS + TURN_LEFTS
ADVANCED_INSTRUCTION_INDICES = LOOP_TWICES + LOOP_THREES + LOOP_FOURS + LOOP_ENDS

ORIENTATION_INDICES = EMPTIES_ORT + ARROWS_DOWN + ARROWS_RIGHT + ARROWS_UP + ARROWS_LEFT
CROSS_INDICES = EMPTIES_CRS + CROSSES


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


if __name__ == "__main__":
    fig1 = plot_samples(BASIC_INSTRUCTION_INDICES, 5, 5)
    fig1.savefig('./output_visualizations/observations_in_basic_instructions.pdf')
    plt.show()

    fig2 = plot_samples(ADVANCED_INSTRUCTION_INDICES, 5, 4)
    fig2.savefig('./output_visualizations/observations_in_advanced_instructions.pdf')
    plt.show()

    fig3 = plot_samples(ORIENTATION_INDICES, 5, 5)
    fig3.savefig('./output_visualizations/observations_in_orientation.pdf')
    plt.show()

    fig4 = plot_samples(CROSS_INDICES, 5, 2)
    fig4.savefig('./output_visualizations/observations_in_crosses.pdf')
    plt.show()
