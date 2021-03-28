# %%
import os
from fastai.vision.all import *
from shutil import rmtree
import nrrd
from scipy.ndimage.interpolation import zoom
from .overlays import insert_image, download, get_center, get_orientation

from . import dataloading
from .video import make_temp_pngs, array_to_png, png_to_video, array_to_video

path = Path(os.path.dirname(__file__))
model_dir = path / 'models'
infer_dir = path / 'inference'

# default_model = sorted(model_dir.glob('*.pkl'), reverse=True)[0]
# default_model = f'{default_model.stem}{default_model.suffix}'
# default_model = '20210307_1002.pkl'
default_model = '20210317_2332_norm.pkl.pth'


def alpha_mask(originals, predictions):
    img = np.copy(originals)
    img[predictions==1, 0] = 255
    img[predictions==2, 1] = 255
    return img


class HedgieFinder():
    def __init__(self, video, model_name=default_model, fps=2, cleanup=True):
        self.video = Path(video)
        print(f"Using model: {model_dir/model_name}")
        self.model = load_learner(model_dir/model_name)
        originals = make_temp_pngs(video, fps=fps) if video.suffix == '.mp4' else video.parent
        self.originals_dir = originals
        self.png_dir = originals.parent
        self.cleanup=cleanup

    def __del__(self):
        if self.cleanup:
            rmtree(self.png_dir)

    def predict(self, fnames=None):
        image_files = fnames or get_image_files(self.originals_dir)
        test_dl = self.model.dls.test_dl(image_files)
        self.originals = np.stack([test_dl.create_item(idx)[0] for idx in test_dl.get_idxs()])
        preds = self.model.get_preds(dl=test_dl, with_decoded=True)[-1]
        zoom_factors = np.array(self.originals.shape[:-1])/np.array(preds.shape)
        self.predictions = zoom(preds, zoom_factors, order=0, prefilter=False)
        return self

    def export_to_video(self, savename=None):
        fname = savename or self.video.parent / f'{Path(self.video).stem}_seg.mp4'
        video_array = alpha_mask(self.originals, self.predictions)
        proc_dir = array_to_png(video_array, self.png_dir / 'processed')
        png_to_video(proc_dir, fname)
        return fname

    def export_to_nrrd(self, savename=None):
        fname = savename or self.video.parent / f'{Path(self.video).stem}'
        seg_name =  str(fname) + '_seg.nrrd'
        original_name = str(fname) + '.nrrd'
        print(self.originals.shape)
        print(self.predictions.shape)
        nrrd.write(seg_name, np.transpose(self.predictions, [0, 2, 1])[:,:,::-1,...]) #flip updown in the frame
        nrrd.write(original_name, np.transpose(self.originals, [0, 2, 1, 3])[:,:,::-1,...])
        return original_name, seg_name


def predict(video, model_name=default_model, savename=None, fps=2):
    return HedgieFinder(video, model_name, fps=fps).predict().export_to_video(savename)


def predict_overlay(video: str, overlay_file: str, savename=None, model_name: str=default_model, fps=2, size: tuple = (None, None), orientation: bool = True):
    """
    Finds the hedgehog in each frame in the `video` and adds an overlay mask given in `overlay_file`

    =====
    Required Args:
    video: filename of video to be process e.g. `myvid.mp4`
    overlay_file: filename of an image to be used as an overlay mask whereever a hedgehog is found
    =====
    Optional Args:
    savename: specify the output video filename, default is the original savename + '_overlay'
    model_name: the fastai pkl file used to segment the hedgehog in each frame
    fps: frames per second, higher fps makes for a larger file size and longer processing, but appears closer to real speed, lower fps is a shorter sped-up version
    size: size of the overlay image in pixels
    orientation: if true the overlay will change orientation to point in the direction of the hedgehog
    """
    hf = HedgieFinder(video, model_name, fps=fps).predict()
    originals, preds = hf.originals, hf.predictions
    if not all(size):
        size = (originals.shape[0]//20, originals.shape[1]//20)
    if orientation:
        overlay_array = (insert_image(o, overlay_file, *get_center(p), sz = size, orientation = get_orientation(p)) for o,p in zip(originals, preds))
    else:
        overlay_array = (insert_image(o, overlay_file, *get_center(p), sz = size) for o,p in zip(originals, preds))
    savename = savename or video.stem + '_overlay.mp4'
    return array_to_video(overlay_array, savename)


def predict_overlay_url(video, overlay_url: str, **kwargs):
    """
    Same as `predict_overlay(video: str, overlay_file: str, savename=None, model_name: str=default_model, fps=2, size: tuple = (200, 200), orientation: bool = True)`
    except that a url pointing to an image is given. The url image is downloaded and then processed using predict_overlay
    """
    overlay_file = download(overlay_url)    
    savename = predict_overlay(video, overlay_file, **kwargs)
    os.remove(overlay_file)
    return savename


if __name__ == '__main__':
    model_name = default_model
    test_video = Path(r'..\videos\test\20210226231222029.mp4')
    predict(model_name, test_video)