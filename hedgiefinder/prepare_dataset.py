# %%
from pathlib import Path
import nrrd
import matplotlib.pyplot as plt
from skimage.morphology import dilation, square
import numpy as np

img_dir = Path(r'data')
mask_dir = Path(r'masks')

masks = list(mask_dir.glob('*.nrrd'))
img_dirs = list(img_dir.glob('*'))
# %%
base_outdir = Path("labels")

outline_width = 15
reg = [];#[[300, 1000],[700, 1500]]

def add_outline(mask, outline_width):
        kern = square(outline_width)
        dilated_mask = dilation(mask, kern)
        outline_mask = 2 * (dilated_mask - mask)
        mask[outline_mask>0] = outline_mask[outline_mask>0]
        return mask

def crop(im, reg):
        return im[reg[0][0]:reg[0][1], reg[1][0]:reg[1][1]]


def process(im):
        im = add_outline(im, outline_width)
        if reg:
                im = crop(im, reg)
        return im


for msk in masks:
    outdir = base_outdir / msk.stem
    outdir.mkdir(exist_ok=True, parents=True)
    print(outdir)
    mask = nrrd.read(msk)[0]
    for i in range(mask.shape[-1]):
        fname = outdir / f"m_{i+1}.png"
        outline_mask = process(mask[..., i].T)
        plt.imsave(fname, outline_mask, cmap='gray')

# %%

