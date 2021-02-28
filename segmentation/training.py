# %%
# https://forums.fast.ai/t/unet-binary-segmentation/29833/36
# https://walkwithfastai.com/Binary_Segmentation
from fastai.vision.all import *
from functools import partial
# %% [markdown]
# images
path = Path(".")
fnames = get_image_files(path/"data")
fnames[0]
# %% [markdown]
# labels
(path/"labels").ls()[0]
# %%

def label_func(fn):
  return path/"labels"/fn.parent.stem/f"m_{fn.stem}{fn.suffix}"


def get_msk(fn):
  "Grab a mask from a `filename` and adjust the pixels based on `pix2class`"
  fn = label_func(fn)
  msk = np.array(PILMask.create(fn))
  levels = np.unique(msk)
  for idx, level in enumerate(levels):
    msk[msk==level] = idx
  return PILMask.create(msk)

get_y = lambda o: get_msk(o)

codes = ['Background', 'Hog', 'Outline']

hogvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_y,
                   item_tfms=Resize(224*2, method='squeeze'))

dls = hogvid.dataloaders(path / "data", path=path, bs=4)
# %%
dls.show_batch(max_n=6)
# %%
learn = unet_learner(dls, resnet34, cbs=ShowGraphCallback())
learn.fine_tune(6)
# %%
learn.show_results(max_n=6, figsize=(7,8))
# %%
import matplotlib.pyplot as plt
out = learn.predict(fnames[0])
# %%
def show_res(fn):
  im = np.array(PILMask.create(fn))
  out = learn.predict(fnames[0])[1]
  f, axs = plt.subplots(1,2, figsize=(8,8))
  axs[0].imshow(im, cmap='gray')
  axs[1].imshow(out)

show_res(fnames[1])
# %%
