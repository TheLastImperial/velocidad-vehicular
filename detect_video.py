import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from collections import deque

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import YoloTF.utils as utils
from YoloTF.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import TLI.utils as TLI_utils
from TLI.tracker import gen_trackers, save_trk


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_boolean('output', False, 'Save the video result')
flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.75, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_multi_integer('c_img', None, 'Cut image by an given size.')
flags.DEFINE_multi_integer('x_lim', None, 'Limits to detections in X axis.')
flags.DEFINE_boolean('time', False, 'Limits to detections in X axis.')
flags.DEFINE_boolean('csv', False, 'Get an csv file with the video precessed.')
flags.DEFINE_integer('max_age', 4, 'Max life time to an tracker')
flags.DEFINE_integer('min_hits', 1, 'Min times to set an tracker')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    root_path = video_path.split("/")
    name = root_path[-1].split(".")[0]
    root_path = "/".join(root_path[:-1])

    seconds_file = None
    if FLAGS.csv:
        seconds_file = TLI_utils.csv_to_list("{}/{}.txt".format(root_path, name))

    tracker_list =[]
    track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
        'S', 'T', 'U', 'V', 'X', 'Y', 'Z',
        'A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1',
        'H1', 'I1', 'J1', 'K1', 'L1', 'M1', 'N1', 'O1', 'P1', 'Q1', 'R1',
        'S1', 'T1', 'U1', 'V1', 'X1', 'Y1', 'Z1'])
    good_tracker_list = []

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    g_vars = {
        "fps": vid.get(cv2.CAP_PROP_FPS),
        "f_count": 1,
        "seconds": seconds_file,
        "file_name": name,
        "root_path": root_path,
        "width": int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "c_img": FLAGS.c_img
    }

    out = None

    if FLAGS.output:
        output_name = "{}/{}_res.mp4".format(root_path, name)
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(output_name, codec, fps, (width, height))

    if FLAGS.time:
        start_time_r = time.time()

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        img_process = np.copy(frame)
        if FLAGS.c_img is not None:
            img_process = TLI_utils.cut_img(frame, FLAGS.c_img)

        frame_size = img_process.shape[:2]
        image_data = cv2.resize(img_process, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        pred_bbox = TLI_utils.post_prediction(pred_bbox, frame_size,
            cut_img= FLAGS.c_img, score= FLAGS.score)


        tracker_list, track_id_list, good_tracker_list = gen_trackers(pred_bbox,
            FLAGS.max_age, FLAGS.min_hits, tracker_list, track_id_list,
            FLAGS.x_lim, g_vars, cut_img=FLAGS.c_img, out_csv=False
        )

        result = np.asarray(frame)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        for trk in good_tracker_list:
            TLI_utils.draw_tracker(result, trk)
            if FLAGS.csv:
                save_trk(trk, tracker_list, result)

        TLI_utils.draw_limits_area(result, FLAGS.x_lim, FLAGS.c_img)

        if FLAGS.time:
            fps = 1.0 / (time.time() - start_time)
            seconds = time.time() - start_time_r

            TLI_utils.set_time_str(result, seconds, pre_txt="Precess Time: ")
            seconds = TLI_utils.get_seconds_from_fps(g_vars["f_count"], g_vars["fps"])
            TLI_utils.set_time_str(result, seconds, pos=(30, 60),
                pre_txt="Real Time: ")
            TLI_utils.set_text(result, "FPS: {:.3f}, Current Frame: {}"
                .format(g_vars["fps"], g_vars["f_count"]), pos=(30, 90)
            )
            TLI_utils.set_text(result, "FPS P: {}".format(fps), pos=(30, 120))



        if not FLAGS.dont_show:
            cv2.imshow("result", result)

        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        g_vars["f_count"] += 1
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
