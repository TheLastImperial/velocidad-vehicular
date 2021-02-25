import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from YOLO.configs import *
from YOLO.utils import Load_Yolo_model
from TLI.tracker import detection

path = "resources/test/20210219.mp4"

yolo = Load_Yolo_model()
detection(path, yolo,
    [500, 1200],
    cut_img=[200, 900, 100, 1500],
    # video_out="test_v.mp4",
    show=True,
    out_csv=True,
    set_time=True)

# from TLI.utils import show_limits

# show_limits(path, [500, 1200], [200, 900, 100, 1500])
