import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import cv2
import numpy as np
import tensorflow as tf


from YOLO.configs import *
from YOLO.utils import Load_Yolo_model
from TLI.utils import detect_video, show_video, get_bboxes

from TLI.tracker import detection

yolo = Load_Yolo_model()
detection("./resources/resize/30_hd.mp4", yolo)
