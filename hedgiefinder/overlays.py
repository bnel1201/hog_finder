import urllib

from PIL import Image
import numpy as np
from skimage.measure import regionprops
from scipy.ndimage.interpolation import zoom


def get_center(mask):
    if mask.sum() > 0:
        props = regionprops(mask)
        center =  props[0].centroid
    else:
        center = (-1, -1)
    return center


def add_overlay(img, overlay, x, y):
    new_img = np.copy(img)
    x, y = int(x), int(y)
    sz = overlay.shape
    mask = np.zeros(img.shape)
    if x > 0 and y > 0: 
        mask[x-sz[0]//2:x+sz[0]//2, y-sz[1]//2:y+sz[1]//2, :] = overlay
        new_img[mask>0] = mask[mask>0]
    return new_img


def insert_image(img, fname, x, y, sz=(200, 200)):
    overlay_array = np.array(Image.open(fname).convert('RGB'))
    z_factors = np.array((*sz, 3))/np.array(overlay_array.shape)
    return add_overlay(img, zoom(overlay_array, z_factors), x, y)


def download(url, fname='temp.png'):
    urllib.request.urlretrieve(url, fname)
    return fname