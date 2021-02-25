# %%
from pathlib import Path
import nrrd
import matplotlib.pyplot as plt


img_dir = Path(r'data')
mask_dir = Path(r'masks')

masks = list(mask_dir.glob('*.nrrd'))
img_dirs = list(img_dir.glob('*'))
# %%
base_outdir = Path("labels")

for msk in masks:
    outdir = base_outdir / msk.stem
    outdir.mkdir(exist_ok=True, parents=True)
    print(outdir)
    mask = nrrd.read(msk)[0]
    for i in range(mask.shape[-1]):
        fname = outdir / f"m_{i+1}.png"
        plt.imsave(fname, mask[..., i].T, cmap='gray')

# %%
