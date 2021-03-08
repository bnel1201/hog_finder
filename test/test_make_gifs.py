from hedgiefinder.video import video_to_gif
import os


def test_make_gif():
    video = r"D:\Dev\hedgehog_finder\data\videos\test\vid1.mp4"
    video_to_gif(video, 'test.gif')
    os.remove('test.gif')
