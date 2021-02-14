# import the necessary packages
from __future__ import print_function
import os
import threading
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image
from PIL import ImageTk
import tkinter as tk
import tkinter.filedialog
import imutils
import cv2


from GUI.utils.video_processing import VideoProcessing
from YOLO.utils import Load_Yolo_model

class VehicleTrackingApp:
    def __init__(self):
        self.__init_vars()
        self.__conf_window()
        self.__set_elements()

    def __init_vars(self):
        self.root = tk.Tk()
        self.thread = None
        self.stop_event = threading.Event()

        self.panel = tk.Label()
        self.lbl_video = tk.Label(self.root, text="Sin video")
        self.btn_play = None
        self.btn_search = None

        # self.video_path = "resources/Testing/original/20200920_2_short.mp4"
        self.video_path = None
        self.vp = None

        self.yolo = Load_Yolo_model()

    def __conf_window(self):
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        # window.resizable(0,0)
        self.root.geometry("{}x{}".format(self.width, self.height))
        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def __set_elements(self):
        self.btn_play = tk.Button(self.root,
            text = "Reproducir",
            command =self.play_video
            )
        self.btn_search = tk.Button(self.root,
            text= "Buscar Video",
            command=self.browseFiles)

        # self.btn_play.pack( side="top" )
        # self.btn_search.pack( side="top" )

        # l1.grid(row = 0, column = 0, sticky = W, pady = 2)
        # l2.grid(row = 1, column = 0, sticky = W, pady = 2)

        self.btn_search.grid(row = 0, column = 0)
        self.lbl_video.grid(row=0, column = 1)

        self.btn_play.grid(row = 1, column = 0)

        # self.panel.config(height=int(self.height * 0.9))
        self.panel.grid(row=2, column = 0)
        # self.panel.pack(side="left")

    def play_video(self):
        if self.video_path is None:
            return

        self.thread = threading.Thread(target=self.vp.video_loop, args=())
        self.thread.start()

    def browseFiles(self):
        self.video_path = tkinter.filedialog.askopenfilename(initialdir = "./",
                                          title = "Selecciona el video",
                                          filetypes = [("Videos", "*.mp4")]
                                          )
        if not self.video_path:
            return

        self.lbl_video.config(text= self.video_path.split("/")[-1])

        self.vp = VideoProcessing(self.panel,
            self.video_path,
            self.stop_event,
            yolo = self.yolo
        )
        self.vp.set_image(self.vp.first_img())

    def test_button(self):
        pritn("Hola mundo")

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stop_event.set()
        self.root.quit()

