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


def get_msk(fn, p2c):
  "Grab a mask from a `filename` and adjust the pixels based on `pix2class`"
  fn = label_func(fn)
  msk = np.array(PILMask.create(fn))
  mx = np.max(msk)
  for i, val in p2c.items():
    msk[msk==i] = val
  return PILMask.create(msk)

p2c = {0: 0, 255: 1}


get_y = lambda o: get_msk(o, p2c)

codes = ['Background', 'Hog']

hogvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_y,
                   item_tfms=Resize(224))

dls = hogvid.dataloaders(path / "data", path=path, bs=8)
# %%
dls.show_batch(max_n=6)
# %%
learn = unet_learner(dls, resnet34)
learn.fine_tune(6)
# %%
learn.show_results(max_n=6, figsize=(7,8))
# %%
out = learn.predict(fnames[0])
# %%
