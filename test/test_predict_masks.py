# %%
from pathlib import Path
import hedgiefinder
from os import remove

def test_predict():
    video = Path(r'D:\Dev\hedgehog_finder\data\videos\test\vid1.mp4')
    video.exists()
    hedgiefinder.predict(video, fps=0.01)
    remove(video.parent / f'{video.stem}_seg{video.suffix}')
# %%
