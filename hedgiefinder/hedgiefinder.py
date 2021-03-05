import sys

import hedgiefinder

video = sys.argv[1]

if len(sys.argv)>2:
    model_name = sys.argv[2]
else:
    model_name = hedgiefinder.inference.default_model

hedgiefinder.predict(video, model_name)
