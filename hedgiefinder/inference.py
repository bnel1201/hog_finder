# %%
import os
from fastai.vision.all import *
from shutil import rmtree
import tempfile, ffmpeg

from . import dataloading

path = Path(os.path.dirname(__file__))
model_dir = path / 'models'
infer_dir = path / 'inference'

default_model = '20210302_2158.pkl'

# %%
# https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#generate-thumbnail-for-video

def make_temp_pngs(test_video, fps=2):
    tempvid = Path(tempfile.gettempdir()) /'hogvid' / 'originals' /r'%d.png'
    tempvid.parent.mkdir(exist_ok=True, parents=True)
    (
        ffmpeg
        .input(test_video)
        .filter('fps', fps=fps, round='up')
        .output(str(tempvid))
        .run()
    )
    return tempvid.parent


def alpha_mask(img, mask):
    out_mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(img.T.shape[1:]))
    img[out_mask==1, 0] = 255
    img[out_mask==2, 1] = 255
    return img


def process_image(model, fname):
    out = model.predict(fname)[1].numpy()
    img = np.array(PILImage.create(fname))
    return alpha_mask(img, out)


def process_video(model, png_dir):
    processed_dir = png_dir.parent / 'processed'
    processed_dir.mkdir(exist_ok=True)
    nfiles = len(png_dir.ls())
    for idx, fname in enumerate(png_dir.ls()):
        img = process_image(model, fname)
        PILImage.create(img).save(processed_dir/f'{idx}.jpg')
        if idx%10 == 0:
            print(f'{idx} / {nfiles}')
    return processed_dir


def export_to_video(processed_dir, fname):
    input_name = str(processed_dir) + f'\%d.jpg'
    output_name = f'{Path(fname).stem}_seg.mp4'
    (
        ffmpeg
        .input(input_name)
        .output(output_name)
        .run()
    )
    return output_name


def predict(test_video, model_name=default_model):
    model = load_learner(model_dir/model_name)
    png_dir = make_temp_pngs(test_video)
    proc_dir = process_video(model, png_dir)
    video_name = f'{Path(test_video).stem}.mp4'
    export_to_video(proc_dir, video_name)
    rmtree(png_dir.parent)
    return video_name

if __name__ == '__main__':
    model_name = default_model
    test_video = Path(r'..\videos\test\20210226231222029.mp4')
    predict(model_name, test_video)