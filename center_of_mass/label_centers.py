# %%
from pathlib import Path
import nrrd
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import pandas as pd

mask_dir = list(Path('../masks').glob('*.nrrd'))
# %%
def get_center(mask):
    if mask.sum() > 0:
        props = regionprops(mask)
        center =  props[0].centroid
    else:
        center = (-1, -1)
    return center
# %%
outdir = Path("com_labels")
outdir.mkdir(exist_ok=True)

for maskfile in mask_dir:
    mask = nrrd.read(maskfile)[0]
    x_list, y_list = [], []
    for i in range(mask.shape[-1]):
        x, y = get_center(mask[...,i])
        x_list.append(x)
        y_list.append(y)
    pd.DataFrame({'x': x_list, 'y': y_list}).to_csv(outdir/f'{maskfile.stem}.csv', index=False)
# %%
