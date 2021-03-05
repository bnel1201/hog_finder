# %%
import os
from fastai.vision.all import *
from shutil import rmtree
import tempfile, ffmpeg
import nrrd
from scipy.ndimage.interpolation import zoom
from numpy import argmax

from . import dataloading

path = Path(os.path.dirname(__file__))
model_dir = path / 'models'
infer_dir = path / 'inference'

default_model = sorted(model_dir.glob('*.pkl'), reverse=True)[0]
default_model = f'{default_model.stem}{default_model.suffix}'

# %%
# https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#generate-thumbnail-for-video

def make_temp_pngs(test_video, fps=2):
    tempvid = Path(tempfile.gettempdir()) /'hogvid' / 'originals' / r'%06d.png'
    tempvid.parent.mkdir(exist_ok=True, parents=True)
    print("Loading Video...")
    (
        ffmpeg
        .input(test_video)
        .filter('fps', fps=fps, round='up')
        .output(str(tempvid))
        .run()
    )
    return tempvid.parent


def video_to_png(video, savedir):
    savedir.mkdir(exist_ok=True)
    for idx, frame in enumerate(video):
        PILImage.create(frame).save(savedir/f'{idx:06}.jpg')
    return savedir


def png_to_video(processed_dir, output_name):
    input_name = str(Path(processed_dir) / r'%06d.jpg')
    print(input_name)
    (
        ffmpeg
        .input(input_name)
        .output(str(output_name))
        .run()
    )
    return output_name


def alpha_mask(originals, predictions):
    img = np.copy(originals)
    img[predictions==1, 0] = 255
    img[predictions==2, 1] = 255
    return img

class HedgieFinder():
    def __init__(self, video, model_name=default_model, fps=2):
        self.video = Path(video)
        self.model = load_learner(model_dir/model_name)
        originals = make_temp_pngs(video, fps=fps)
        self.originals_dir = originals
        self.png_dir = originals.parent

    def __del__(self):
         rmtree(self.png_dir)

    def predict(self):
        test_dl = self.model.dls.test_dl(get_image_files(self.originals_dir))
        self.originals = np.stack([test_dl.create_item(idx)[0] for idx in test_dl.get_idxs()])
        preds = self.model.get_preds(dl=test_dl, with_decoded=True)[-1]
        zoom_factors = np.array(self.originals.shape[:-1])/np.array(preds.shape)
        self.predictions = zoom(preds, zoom_factors, order=0, prefilter=False)
        return self

    def export_to_video(self, savename=None):
        video = alpha_mask(self.originals, self.predictions)
        proc_dir = video_to_png(video, self.png_dir / 'processed')
        video_name = savename or self.video.parent / f'{Path(self.video).stem}_seg.mp4'
        png_to_video(proc_dir, video_name)
        return video_name

    def export_to_nrrd(self, savename=None):
        fname = savename or self.video.parent / f'{Path(self.video).stem}'
        seg_name =  str(fname) + '_seg.nrrd'
        original_name = str(fname) + '.nrrd'
        print(self.originals.shape)
        print(self.predictions.shape)
        nrrd.write(seg_name, np.transpose(self.predictions, [0, 2, 1])[:,:,::-1,...]) #flip updown in the frame
        nrrd.write(original_name, np.transpose(self.originals, [0, 2, 1, 3])[:,:,::-1,...])
        return original_name, seg_name


def predict(video, model_name=default_model, fps=2):
    return HedgieFinder(video, model_name, fps=fps).predict().export_to_video()


if __name__ == '__main__':
    model_name = default_model
    test_video = Path(r'..\videos\test\20210226231222029.mp4')
    predict(model_name, test_video)