import os
from os import listdir
from threading import Thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from YOLO.utils import Load_Yolo_model
from TLI.tracker import detection

yolo = Load_Yolo_model()

paths = [
    "resources/test/1/20210219.mp4",
    "resources/test/2/20210219.mp4"
]

for path in paths:
    thread = Thread(target = detection, args=(path, yolo, [500, 1200]),
        kwargs = {
            "cut_img":[200, 900, 100, 1500],
            "show": False,
            "out_csv": True
        }).start()
