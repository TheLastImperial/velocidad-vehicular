import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from YOLO.configs import *
from YOLO.utils import Load_Yolo_model
from TLI.utils import detect_video, show_video

yolo = Load_Yolo_model()
# detect_video(yolo, "./imgs/resize/30.mp4", '',
#     input_size=YOLO_INPUT_SIZE, show=True,
#     CLASSES=YOLO_COCO_CLASSES, rectangle_colors=(255,0,0), jump_frames=0)

# show_video("./imgs/resize/30.mp4")

