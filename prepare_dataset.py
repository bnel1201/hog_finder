# %%
from pathlib import Path
import nrrd
import matplotlib.pyplot as plt


img_dir = Path(r'data')
mask_dir = Path(r'masks')

masks = mask_dir.rglob('*.nrrd')
imgs = img_dir.rglob('*.png')

mask = nrrd.read(list(masks)[0])[0]
# %%
outdir = Path("labels")
outdir.mkdir(exist_ok=True)

for i in range(mask.shape[-1]):
    fname = outdir / f"m_{i+1}.png"
    plt.imsave(fname, mask[..., i].T, cmap='gray')

# %%
