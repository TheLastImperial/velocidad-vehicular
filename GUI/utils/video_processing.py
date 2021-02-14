from collections import deque

import cv2
from PIL import Image, ImageTk
import tkinter as tk

from TLI.tracker import detector

class VideoProcessing():
    def __init__(self, panel, video_path, stop_event, yolo=None):
        self.panel = panel
        self.video_path = video_path
        self.stop_event = stop_event
        self.yolo = yolo

    def video_loop(self):
        try:
            vid = cv2.VideoCapture(self.video_path)
            ret, img = vid.read()
            while not self.stop_event.is_set() and ret:
                self.set_image(img)
                ret, img = vid.read()
        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def set_image(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if self.panel is None:
            self.panel = tk.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left")

        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def first_img(self):
        vid = cv2.VideoCapture(self.video_path)
        ret, img = vid.read()
        return img

    def detector(self):
        max_age = 4
        min_hits = 1
        tracker_list =[]
        track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K'])

        cut_img=[200, 900, 100, 1500]
        left, right, top, bottom = cut_img[0], cut_img[1], cut_img[2], cut_img[3]

        x_limits = [500, 1200]

        try:
            vid = cv2.VideoCapture(self.video_path)
            fps = vid.get(cv2.CAP_PROP_FPS)
            ret, img = vid.read()
            while not self.stop_event.is_set() and ret:

                img = detector(img, self.yolo, max_age, min_hits,
                    tracker_list, track_id_list, x_limits, fps,
                    cut_img)
                if x_limits is not None:
                    cv2.line(img,(x_limits[0], 0),
                        (x_limits[0], img.shape[0]),
                        (255,0,0),1)

                    cv2.line(img,(x_limits[1], 0),
                        (x_limits[1], img.shape[0]),
                        (255,0,0),1)

                if cut_img is not None:
                    cv2.rectangle(img, (top, left),
                        (bottom, right),
                        (0, 0, 255), 1)
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                if self.panel is None:
                    self.panel = tk.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left")

                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
                ret, img = vid.read()
                # print("W: {}, H: {}".format(
                #     self.panel.winfo_width(),
                #     self.panel.winfo_height())
                # )
        except RuntimeError:
            print("[INFO] caught a RuntimeError")

