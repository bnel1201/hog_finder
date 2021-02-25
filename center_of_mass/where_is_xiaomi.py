# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
# %%
coords_dir = Path('labels')
images_dir = Path('data')
# %%
def get_images_and_labels(idx=1):
    return list((images_dir/f'{idx}').glob('*')), pd.read_csv(coords_dir/f'{idx}.csv')

# %%
def show_xiaomi(im, labels):
    if len(im)>1:
        im = im[0]

    f, axs= plt.subplots(figsize=(10, 8))
    axs.imshow(plt.imread(im), cmap='gray')
    axs.scatter(labels.x, labels.y, color='r', alpha=0.1)
    axs.axes.xaxis.set_visible(False)
    axs.axes.yaxis.set_visible(False)
    return axs
# %%
output_dir = Path('xiaomi_maps')
output_dir.mkdir(exist_ok=True)

for ind in images_dir.glob('*'):
    ind = ind.stem
    ims, labels = get_images_and_labels(ind)
    show_xiaomi(ims, labels)
    plt.savefig(output_dir/f'xiaomi_map_{ind}.png')
# %%
