from fastai.vision.all import *

path = Path('.').absolute().parent.parent/'data'/'train' #need a better solution than this

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