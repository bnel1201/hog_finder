import sys

import hedgiefinder

video = sys.argv[1]

if len(sys.argv)>2:
    model_name = sys.argv[2]
else:
    model_name = '20210302_2158.pkl'

hedgiefinder.predict(video, model_name)
