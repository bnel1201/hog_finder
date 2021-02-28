# Hog Finder

![finder gif](hog_finder.gif)
## Making dataset

- mp4 to png done with [video to png](video_to_png.bat)
- segmentations done with [slicer](https://www.slicer.org/)

## [model training](segmentation/segmentation_train.ipynb)

## [model inference](segmentation/inference.py)

- turned into a command line program using [hogfinder](segmentation/hogfinder.py):

```bash
python segmentation/hogfinder.py path/to/hedgehog_video.mp4
```