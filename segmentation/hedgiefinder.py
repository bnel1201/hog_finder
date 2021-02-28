import sys

from inference import run_all

video = sys.argv[1]

if len(sys.argv)>2:
    model_name = sys.argv[2]
else:
    model_name = '20210228_1514.pkl'

run_all(model_name, video)