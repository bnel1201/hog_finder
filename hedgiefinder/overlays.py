import urllib

from PIL import Image
import numpy as np
from skimage.measure import regionprops
from scipy.ndimage.interpolation import zoom


def get_center(mask):
    if mask.sum() < 1:
        return (-1, -1)
    props = regionprops(mask)
    return props[0].centroid


def get_orientation(mask):
    "By default region props orientation points along minor axis, so negate"
    if mask.sum() < 1:
        return 0
    props = regionprops(mask)
    return -props[0].orientation


def add_overlay(img, overlay, x, y):
    new_img = np.copy(img)
    x, y = int(x), int(y)
    sz = overlay.shape
    mask = np.zeros(img.shape)
    if x > 0 and y > 0: 
        xmin, ymin = max(x-sz[0]//2, 0), max(y-sz[1]//2, 0)
        mask[xmin:xmin+overlay.shape[0], ymin:ymin+overlay.shape[1], :] = overlay
        new_img[mask>0] = mask[mask>0]
    return new_img


def insert_image(img, fname, x, y, sz=(200, 200), orientation=None):
    overlay_image = Image.open(fname).convert('RGB')
    if orientation:
        overlay_image = overlay_image.rotate(np.rad2deg(orientation), expand=True)
    overlay_array = np.array(overlay_image)
    z_factors = np.array((*sz, 3))/np.array(overlay_array.shape)
    return add_overlay(img, zoom(overlay_array, z_factors), x, y)


def download(url, fname='temp.png'):
    urllib.request.urlretrieve(url, fname)
    return fname