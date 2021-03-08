import os
from pathlib import Path
import tempfile
from fastai.vision.all import PILImage

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


def video_to_gif(video, fname=None):
    savename = fname or Path(video).stem + '.gif'
    cmd = f"""ffmpeg -ss 30 -t 3 -i {video} -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {savename}"""
    os.system(cmd)
    return fname