# %%
import os
from fastai.vision.all import *
from shutil import rmtree
import tempfile, ffmpeg

import dataloading

path = Path(os.path.dirname(__file__))
model_dir = path / 'models'
infer_dir = path / 'inference'

# %%
# https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#generate-thumbnail-for-video

def make_temp_pngs(test_video):
    tempvid = Path(tempfile.gettempdir()) /'hogvid' / 'originals' /r'%d.png'
    tempvid.parent.mkdir(exist_ok=True, parents=True)

    stream = ffmpeg.input(test_video)
    stream = ffmpeg.filter(stream, 'fps', fps=2, round='up')
    stream = ffmpeg.output(stream, str(tempvid))
    ffmpeg.run(stream)
    return tempvid.parent


def process_image(model, fname):
    in_img = np.array(PILImage.create(fname))
    out = model.predict(fname)[1].numpy()
    out_mask = np.array(Image.fromarray(out.astype(np.uint8)).resize(in_img.T.shape[1:]))
    in_img[out_mask==1, 0] = 255
    in_img[out_mask==2, 1] = 255
    return in_img


def process_video(model, png_dir):
    processed_dir = Path(tempfile.gettempdir()) /'hogvid' / 'processed'
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
    # cmd = f"ffmpeg.exe -i {input_name} -r 1 -vcodec mpeg4 -y {output_name}"
    # print(cmd)
    # os.system(cmd)
    (
        ffmpeg
        .input(input_name)
        .output(output_name)
        .run()
    )
    return output_name


def run_all(model_name, test_video):
    model = load_learner(model_dir/model_name)
    png_dir = make_temp_pngs(test_video)
    proc_dir = process_video(model, png_dir)
    video_name = f'{Path(test_video).stem}.mp4'
    export_to_video(proc_dir, video_name)
    rmtree(png_dir.parent)
    return video_name

if __name__ == '__main__':
    model_name = '20210228_1123.pkl'
    test_video = Path(r'..\videos\test\20210226231222029.mp4')
    run_all(model_name, test_video)