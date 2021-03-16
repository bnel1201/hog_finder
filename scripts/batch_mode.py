# %%
from pathlib import Path
import random

from fastai.callback.core import FetchPredsCallback
from hedgiefinder import predict_overlay_url, predict

model = '20210307_1002.pkl'

video_dir = Path(r'D:\Dev\Datasets\xiaomi videos')

in_dir = video_dir / 'originals'
out_dir = video_dir / 'processed'

emojis = [
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/openmoji/272/sparkles_2728.png',
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/openmoji/272/hedgehog_1f994.png',
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/openmoji/272/revolving-hearts_1f49e.png',
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/openmoji/272/smiling-face-with-hearts_1f970.png',
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/openmoji/272/robot_1f916.png',
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/openmoji/272/rocket_1f680.png',
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/mozilla/36/pile-of-poo_1f4a9.png',
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/openmoji/272/balloon_1f388.png',
    'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/openmoji/272/hibiscus_1f33a.png'
]
# %%
fps = 4
for vid in in_dir.glob('*.mp4'):
    emoji = random.choice(emojis)
    overlayname = out_dir / f'{vid.stem}_overlay{vid.suffix}'
    segname = out_dir / f'{vid.stem}_seg{vid.suffix}'
    predict(vid, model_name=model, fps=fps, savename=segname)
    print(predict_overlay_url(vid, emoji, model_name=model, savename=overlayname, fps=fps))
# %%
