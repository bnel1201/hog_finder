# %%
from pathlib import Path
from hedgiefinder import HedgieFinder

video = Path(r'D:\Dev\hedgehog_finder\data\videos\test\vid1.mp4')
# %%
video.exists()
# %%
hf = HedgieFinder(video, fps=0.1).predict()
# %%
hf.export_to_nrrd()

