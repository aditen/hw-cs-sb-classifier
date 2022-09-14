import math

from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data_loading import DataloaderKinderlabor

if __name__ == "__main__":
    df = DataloaderKinderlabor.raw_df()
    classes = df['class'].unique()
    print(f'classes currently in df: {classes}')

    task_types = df['type'].unique().tolist()
    print(f'task types: {task_types}')

    for task_type in task_types:
        if type(task_type) == float and math.isnan(task_type):
            continue
        uk_df = df[(df['type'] == task_type) & (df['label'] == 'NOT_READABLE')]
        print(f'Unknowns for task type {task_type}: {len(uk_df)}')

        ims = []
        for i, _ in uk_df.iterrows():
            ims.append(Image.open(f'{DataloaderKinderlabor.IMG_CSV_FOLDER}{i}.jpeg'))

        n_imgs_to_display = int(math.floor(len(ims) / 4) * 4)
        print(f'Displaying {n_imgs_to_display} out of {len(ims)} images')
        ims = ims[:n_imgs_to_display]

        fig = plt.figure(figsize=(4., max(n_imgs_to_display / 24, 4.)))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(int(n_imgs_to_display / 8), 8),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, ims):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
        plt.savefig(f'output_visualizations/unknowns_{task_type}.pdf')
        plt.show()
