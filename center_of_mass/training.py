# %%
from fastai.vision.all import *
import pandas as pd


path = Path('data')
(path/'1').ls()
# %%
img_files = get_image_files(path)
def img2pose(x): return Path('labels')/f'{x.parent.stem}.csv'
img2pose(img_files[0])
# %%
im = PILImage.create(img_files[0])
im.shape
# %%
im.to_thumb(160)
# %%
def get_ctr(f):
    df = pd.read_csv(img2pose(f))
    idx = int(f.stem)-1
    x = df.x[idx]
    y = df.y[idx]
    return tensor([x,y])
# %%
shrink_factor = 4
hog = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter = RandomSplitter(),
    item_tfms=Resize((1080//shrink_factor, 1920//shrink_factor), method='squeeze')
)
# %%
dls = hog.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8,6))
# %%
learn = cnn_learner(dls, resnet18, y_range=(-1,1), cbs=ShowGraphCallback())
# %%
# learn.lr_find()
# %%
learn.fine_tune(50, 1e-3)
# %%
learn.show_results()
# %%
learn.show_training_loop()
# %%