from fastai.vision.all import *

def label_func(fn):
  return fn.parents[2]/'labels'/fn.parents[0].stem/f"m_{fn.stem}{fn.suffix}"


def get_msk(fn):
  fn = label_func(fn)
  msk = np.array(PILMask.create(fn))
  levels = np.unique(msk)
  for idx, level in enumerate(levels):
    msk[msk==level] = idx
  return PILMask.create(msk)