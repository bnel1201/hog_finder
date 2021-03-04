import matplotlib.pyplot as plt
import nrrd
from skimage.morphology import dilation, square
import numpy as np

reg = [];#[[300, 1000],[700, 1500]]
outline_width = 15

def add_outline(mask, outline_width):
        kern = square(outline_width)
        mask[mask > 0] = 1
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


def save_images(image_file, outdir):
    image = nrrd.read(image_file)[0]
    for idx, im in enumerate(image):
        fname = outdir / f"{idx+1}.png"
        plt.imsave(fname, np.flipud(np.transpose(im, [1,0,2])))


def save_labels(mask_file, outdir):
    mask = nrrd.read(mask_file)[0]
    for idx, msk in enumerate(mask):
        fname = outdir / f"m_{idx+1}.png"
        outline_mask = process(np.flipud(msk.T))
        plt.imsave(fname, outline_mask, cmap='gray')
