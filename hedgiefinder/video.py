import os
from pathlib import Path
import tempfile
from fastai.vision.all import PILImage
from shutil import rmtree
from os import remove

import ffmpeg


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


def array_to_png(video, savedir):
    savedir.mkdir(exist_ok=True)
    for idx, frame in enumerate(video):
        PILImage.create(frame).save(savedir/f'{idx:06}.jpg')
    return savedir


def png_to_video(processed_dir, output_name, cleanup=False):
    input_name = str(Path(processed_dir) / r'%06d.jpg')
    print(input_name)
    (
        ffmpeg
        .input(input_name)
        .output(str(output_name))
        .overwrite_output()
        .run()
    )
    if cleanup:
        rmtree(processed_dir)
    return output_name


def array_to_video(array, videoname):
    tempdir = Path(tempfile.mkdtemp())
    array_to_png(array, tempdir)
    return png_to_video(tempdir, videoname, cleanup=True)


def array_to_gif(array, gifname):
    vid = array_to_video(array, gifname)
    gif = video_to_gif(vid, gifname)
    remove(vid)
    return gif


def video_to_gif(video, fname=None):
    savename = fname or Path(video).stem + '.gif'
    cmd = f"""ffmpeg -ss 30 -t 3 -i {video} -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {savename}"""
    os.system(cmd)
    return fname