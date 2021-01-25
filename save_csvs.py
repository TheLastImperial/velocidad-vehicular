import os
from os import listdir
from threading import Thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from YOLO.utils import Load_Yolo_model
from TLI.tracker import detection


yolo = Load_Yolo_model()

root_path = "resources/Testing/"
input_path = "original/"
output_path = "output/"


for f in listdir(root_path + input_path):
    cvs_out = f.split(".")[0] + ".csv"

    inp = root_path + input_path + f
    out = root_path + output_path + cvs_out

    thread = Thread(target = detection, args=(inp, yolo),
        kwargs = {
            "cut_img":[200, 900, 100, 1500],
            "show": False,
            "out_csv": out
        }).start()

