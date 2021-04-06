import os

# command = ["python detect_video.py --weights model_data/yolov3-416",
# "--model yolov3 --video {} --c_img {} --c_img {} --c_img {} --c_img {}",
# "--x_lim {} --x_lim {} --time True --csv True"]


command = ["python detect_video.py --weights model_data/yolov3-416",
"--model yolov3 --video {}",
"--x_lim {} --x_lim {} --time True --csv True --dont_show True --output True"]

c_img = [200, 900, 100, 1500]
x_lim = [500, 1200]

paths = [
    "resources/20200302/10/VID_20210302_095247.mp4",
    "resources/20200302/11/VID_20210302_095602.mp4",
    "resources/20200302/12/VID_20210302_095653.mp4",
    "resources/20200302/13/VID_20210302_095705.mp4",
    "resources/20200302/14/VID_20210302_095850.mp4",
    "resources/20200302/15/VID_20210302_100133.mp4",
    "resources/20200302/16/VID_20210302_100408.mp4",
    "resources/20200302/17/VID_20210302_100616.mp4",
    "resources/20200302/18/VID_20210302_100702.mp4",
    "resources/20200302/19/20210219.mp4",
    "resources/20200302/1/VID_20210302_094109.mp4",
    "resources/20200302/2/VID_20210302_094504.mp4",
    "resources/20200302/3/VID_20210302_094544.mp4",
    "resources/20200302/4/VID_20210302_094644.mp4",
    "resources/20200302/5/VID_20210302_094741.mp4",
    "resources/20200302/6/VID_20210302_094820.mp4",
    "resources/20200302/7/VID_20210302_094841.mp4",
    "resources/20200302/8/VID_20210302_095100.mp4",
    "resources/20200302/9/VID_20210302_095158.mp4"
]

# paths = [
#     "resources/20200302/1/VID_20210302_094109.mp4",
# ]


for p in paths:
    r_command = " ".join(command).format(p, x_lim[0], x_lim[1])
    os.system(r_command)

