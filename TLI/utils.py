import time

import cv2
import numpy as np
import tensorflow as tf

from YOLO.configs import *
from YOLO.utils import image_preprocess, postprocess_boxes, nms, draw_bbox


def detect_video(Yolo, video_path, output_path, input_size=416,
    show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3,
    iou_threshold=0.45, rectangle_colors='', jump_frames=4):

    times, times_2 = [], []
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    frames = jump_frames
    count = 0
    while True:
        count+=1
        _, img = vid.read()



        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break
        if frames == jump_frames:
            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            t1 = time.time()

            pred_bbox = Yolo.predict(image_data)

            t2 = time.time()

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')

            image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)


            t3 = time.time()
            times.append(t2-t1)
            times_2.append(t3-t1)

            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

            # image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
            frames = 0
            print("Count {}".format(str(count)))
        else:
            image = original_image
            frames+=1
            print("Count {}".format(str(count)))

        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                vid.release()
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()


#############################################################################
# This method get the bboxes, scores and classes.
# The BBoxes have the format:
# [[top, left, bottom, right], ...]
def get_bboxes(Yolo, img, input_size=416, score_threshold=0.3, iou_threshold=0.45):
    image_data = image_preprocess(np.copy(img), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = Yolo.predict(image_data)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, img, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    if len(bboxes) == 0:
        return np.array([]), np.array([]), np.array([])

    transpose = np.transpose(bboxes)
    coors = np.array(transpose[:4], dtype=np.int32)

    scores = transpose[4]
    classes = transpose[5].astype(np.int32)

    return np.transpose(coors), scores, classes

def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)

    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1)

        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x= 'x='+str((left+right)/2)
        cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y= 'y='+str((top+bottom)/2)
        cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)

    return img

def draw_all_boxes(img, bboxes, box_color=(255, 0, 0)):
    for box in bboxes:
        left, top, right, bottom = box[1], box[0], box[3], box[2]
        cv2.rectangle(img, (left, top), (right, bottom), box_color, 1)

def move_x_to_y(coors):
    coors2 = coors.T
    if len(coors2) != 4:
        return coors
    coors2 = np.array([coors2[1], coors2[0], coors2[3], coors2[2]])
    coors2 = coors2.T
    return coors2

def draw_tracker(img, tracker, box_color=(0, 255, 255), track_color=(0, 255, 255), show_box=True, show_track=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = tracker.box[1], tracker.box[0], tracker.box[3], tracker.box[2]

    if show_box:
        # Draw the bounding box
        cv2.rectangle(img, (left, top), (right, bottom), box_color, 1)

    if show_track:
        # Draw track
        pts = np.array(tracker.j_track(), np.int32)
        if pts.size > 1:
            cv2.polylines(img, [pts], False, box_color)

        pts = np.array(tracker.straight, np.int32)
        if pts.size > 1:
            cv2.polylines(img, [pts], False, (0, 0, 255))
    return img
def show_video(video_path):
    vid = cv2.VideoCapture(video_path)
    ret, img = vid.read()
    while ret:

        cv2.imshow('Testing', img)
        ret, img = vid.read()
        if cv2.waitKey(25) & 0xFF == ord("q"):
            vid.release()
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()

def get_seconds_from_fps(current_frame, fps):
    return current_frame / float(fps)

def set_time_str(img, seconds, font = cv2.FONT_HERSHEY_SIMPLEX,
    font_size = 0.7, font_color = (0, 0, 0), pos=(30, 30)):

    mils = "000" + str(int(seconds * 1000))
    secs = "00" + str(int(seconds%60))
    mins = "00" + str(int(seconds/60))
    time_str = "{}:{}:{}".format(mins[-2:], secs[-2:], mils[-3:])

    set_text(img, time_str, font = font,
        font_size = font_size, font_color = font_color, pos=pos)

def set_text(img, txt, font = cv2.FONT_HERSHEY_SIMPLEX,
    font_size = 0.7, font_color = (0, 0, 0), pos=(30, 30)):
    cv2.putText(img, txt, pos,
                font, font_size, font_color, 1, cv2.LINE_AA)

def csv_to_list(path):
    result = []
    with open(path, "r") as f:
        for line in f:
            result.append(list(map(float, line.split(","))))
    return result

def show_limits(path, x_limits, cut_img):
    vid = cv2.VideoCapture(path)
    _, img = vid.read()

    # x_limits = [500, 1200]
    # cut_img=[200, 900, 100, 1500]
    left, right, top, bottom = cut_img[0], cut_img[1], cut_img[2], cut_img[3]
    cv2.line(img,(x_limits[0], 0),(x_limits[0], img.shape[0]),
        (255,0,0),1)
    cv2.line(img,(x_limits[1], 0),(x_limits[1], img.shape[0]),
        (255,0,0),1)
    cv2.rectangle(img, (top, left),(bottom, right), (0, 0, 255), 1)

    cv2.imshow("Limits", img)
    cv2.waitKey(0)
    vid.release()
