# -*- coding: utf-8 -*-
'''
Implement and test tracker
Source: https://awesomeopensource.com/project/kcg2015/Vehicle-Detection-and-Tracking
'''
from collections import deque

import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag
from scipy.optimize import linear_sum_assignment
import cv2

from TLI import utils

class Tracker(): # class for Kalman Filter-based tracker
    def __init__(self, x_limits, g_vars):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id

        # Format to box: top, left, bottom, right
        self.box = [] # list to store the coordinates for a bounding box
        self.hits = 0 # number of detection matches
        self.no_losses = 0 # number of unmatched tracks (track loss)

        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.x_state=[]
        self.dt = 1.   # time interval

        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, self.dt, 0,  0,  0,  0,  0, 0],
                           [0, 1,  0,  0,  0,  0,  0, 0],
                           [0, 0,  1,  self.dt, 0,  0,  0, 0],
                           [0, 0,  0,  1,  0,  0,  0, 0],
                           [0, 0,  0,  0,  1,  self.dt, 0, 0],
                           [0, 0,  0,  0,  0,  1,  0, 0],
                           [0, 0,  0,  0,  0,  0,  1, self.dt],
                           [0, 0,  0,  0,  0,  0,  0,  1]])

        # Measurement matrix, assuming we can only measure the coordinates

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])


        # Initialize the state covariance
        self.L = 10.0
        self.P = np.diag(self.L*np.ones(8))


        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt**4/4., self.dt**3/2.],
                                    [self.dt**3/2., self.dt**2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)

        # Initialize the measurement covariance
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L,
            self.L, self.L])
        self.R = np.diag(self.R_diag_array)


        # Attributes to draw an line track and know if the increase
        # or decree the area size.
        self.center = np.array([])
        self.w = 0
        self.h = 0
        self.area = 0
        self.areas = []
        self.track = []
        self.rocs = []
        self.roc = 0
        self.straight = [[]]
        self.straights = []
        self.dimensions = []
        self.angle = []
        self.angles = []
        self.dimension = []

        self.boxes = []

        self.class_id = None
        self.score = None

        self.x_limits = x_limits
        self.limits_ind = [None, None]
        self.g_vars = g_vars
        self.saved = [False, False]
        self.imgs = []

    def update_R(self):
        R_diag_array = self.R_scaler * np.array([self.L, self.L,
            self.L, self.L])
        self.R = np.diag(R_diag_array)

    def add_box(self, box):
        self.box = box
        self.boxes.append(box)

        top, left, bottom, right = box[0], box[1], box[2], box[3]

        # Center format [x, y]
        self.center = np.array([
            left + int((right-left)/2), top + int((bottom-top)/2)
        ])
        self.w = right-left
        self.h = bottom-top
        self.area = self.h * self.w
        self.areas.append(self.area)
        self.track.append(self.center)


        # Rate of change
        if len(self.areas) <= 2:
            self.roc = 0
        else:
            ant = self.areas[-2]
            act = self.area
            self.roc = float(ant - act) / (float(ant) + 0.000001)
        self.rocs.append(self.roc)

        self.straight, self.angle, self.dimension = self.gen_straight()
        self.straights.append(self.straight)
        self.angles.append(self.angle)
        self.dimensions.append(self.dimension)

        # if self.x_limits is not None:
        if self.limits_ind[0] is None and self.center[0] >= self.x_limits[0]:
            self.limits_ind[0] = len(self.areas) - 1
        elif self.limits_ind[1] is None and self.center[0] >= self.x_limits[1]:
            self.limits_ind[1] = len(self.areas) - 1

    # This method return the track but with a jump intervale.
    # For example:
    # [[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14]]
    # j_track(jump=2) will return
    # [[1,2], [7,8], [13,14]]
    def j_track(self, jump=10):
        if jump==0:
            return self.track
        track = np.array(self.track)
        t = np.arange(0, len(track))
        t2 = np.arange(0, len(track), jump)
        t = np.delete(t, t2)
        result = np.delete(track, t, axis=0)
        result = np.append(result, [track[-1]], axis=0)
        return result

    # This method generate the straight and the angles to
    # the current track.
    def gen_straight(self, limits = None):
        if limits is None:
            track = np.array(self.track)
        else:
            track = np.array(self.track[limits[0]:limits[1]])
        if(len(track) == 0):
            return np.array([0]), [0, 0]
        x = track.T[0]
        y = track.T[1]

        exy = np.sum(x*y)
        ex = np.sum(x)
        ey = np.sum(y)

        exp = np.sum(np.power(x, 2))

        m = (len(x)*exy) - (ex * ey)

        m /= ((len(x)*exp) - pow(ex, 2)) + 0.000001

        b = np.average(y) - (m * np.average(x))

        y_line = b + (m*x)

        straight = []
        straight.append(x)
        straight.append(np.array([int(y) for y in y_line]))
        straight = np.array(straight).T

        A = 0.0
        B = 0.0
        c1, c2, h = 0, 0, 0
        import math

        if len(x) > 1:
            pt1 = straight[0]
            pt2 = straight[-1]

            A = math.atan((pt1[0] - pt2[0])/(pt1[1] - pt2[1] + 0.0000001))
            A = math.degrees(A)
            straight = np.array([pt1, pt2])
            B = 90.0 - A


            c1 = pt2[1] - pt1[1]
            c2 = pt2[0] - pt1[0]

            h = int(math.sqrt(c1**2 + c2**2))

        return np.array(straight), [A, B], [c1, c2, h]

    def kalman_filter(self, z):
        '''
        Implement the Kalman Filter, including the predict and the update
        stages,with the measurement z
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q

        #Update
        S = dot(self.H, self.P).dot(self.H.T) + self.R
        K = dot(self.P, self.H.T).dot(inv(S)) # Kalman gain
        y = z - dot(self.H, x) # residual
        x += dot(K, y)
        self.P = self.P - dot(K, self.H).dot(self.P)
        self.x_state = x.astype(int) # convert to integer coordinates
                                     #(pixel values)

    def predict_only(self):
        '''
        Implment only the predict stage. This is used for unmatched detections
        and unmatched tracks
        '''
        x = self.x_state
        # Predict
        x = dot(self.F, x)
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q
        self.x_state = x.astype(int)

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched
    detections, unmatchted trackers, unmatched detections.
    '''

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk)
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = utils.box_iou2(trk, det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    indices = linear_sum_assignment(-IOU_mat)
    indices = np.asarray(indices)
    matched_idx = np.transpose(indices)

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return (matches, np.array(unmatched_detections),
            np.array(unmatched_trackers))

# tracker_list: Es una lista con objetos de tipo Tracker
def gen_trackers(prediction, max_age, min_hits, tracker_list, track_id_list,
    x_limits, g_vars, cut_img=None, out_csv=False
    ):
    # Contine solo las bbox de los objetos Tracker.
    bboxes, scores, classes = prediction
    x_box =[]

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(
            x_box, bboxes, iou_thrd = 0.3
        )

    # Deal with matched detections
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = bboxes[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.score = scores[det_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.add_box(xx)
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = bboxes[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Tracker(x_limits=x_limits, g_vars=g_vars) # Create a new tracker
            tmp_trk.class_id = classes[idx]
            tmp_trk.score = scores[idx]
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.add_box(xx)
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.add_box(xx)
            x_box[trk_idx] = xx


    # The list of tracks to be annotated
    good_tracker_list =[]
    for trk in tracker_list:
        # print("Classes: {}, TrkH: {}, Min: {}, TrkN: {}, Max: {}".format(
        #    str(len(classes)), trk.hits, min_hits, trk.no_losses, max_age))
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)

    for trk in deleted_tracks:
            track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    return tracker_list, track_id_list, good_tracker_list

# tracker_list: Es una lista con objetos de tipo Tracker
def detector(img, yolo, max_age, min_hits, tracker_list, track_id_list,
    x_limits, g_vars, cut_img=None, out_csv=False
    ):

    img_process = np.copy(img)
    if cut_img is not None:
        img_process = img[cut_img[0]:cut_img[1], cut_img[2]:cut_img[3]]

    coors, scores, classes = utils.get_bboxes(yolo, img_process,
        score_threshold=0.7)
    z_box = utils.move_x_to_y(coors)

    if cut_img is not None and len(z_box) > 0:
        plus_cut = z_box.T
        plus_cut[0] += cut_img[0]
        plus_cut[1] += cut_img[2]
        plus_cut[2] += cut_img[0]
        plus_cut[3] += cut_img[2]
        z_box = plus_cut.T

    # Contine solo las bbox de los objetos Tracker.
    x_box =[]

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)

    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(
            x_box, z_box, iou_thrd = 0.3
        )

    # Deal with matched detections
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.score = scores[det_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.add_box(xx)
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0

    # Deal with unmatched detections
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = Tracker(x_limits=x_limits, g_vars=g_vars) # Create a new tracker
            tmp_trk.class_id = classes[idx]
            tmp_trk.score = scores[idx]
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.add_box(xx)
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)

    # Deal with unmatched tracks
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.add_box(xx)
            x_box[trk_idx] = xx


    # The list of tracks to be annotated
    good_tracker_list =[]
    for trk in tracker_list:
        # print("Classes: {}, TrkH: {}, Min: {}, TrkN: {}, Max: {}".format(
        #    str(len(classes)), trk.hits, min_hits, trk.no_losses, max_age))
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             img = utils.draw_tracker(img, trk)
             if out_csv:
                save_trk(trk, tracker_list, img=img)
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)

    for trk in deleted_tracks:
            track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    return img

def save_trk(trk, tracker_list, img=None, save_csv=False):

    if trk.limits_ind[0] is not None and \
            trk.limits_ind[1] is None \
        and not trk.saved[0] and img is not None:
        straight1 = trk.straights[trk.limits_ind[0]]
        angle1 = trk.angles[trk.limits_ind[0]]

        if len(straight1) != 2:
            return

        img_w = np.copy(img)
        utils.draw_triangule(straight1, img_w)
        utils.draw_all_boxes(img_w, [trk.box], box_color=(255, 255, 255))
        utils.draw_limits_area(img_w, trk.x_limits, trk.g_vars["c_img"])
        trk.imgs.append(img_w)
        trk.saved[0] = True

    # Garantiza que ya se tenga la imagen de entrada y salida.
    if trk.limits_ind[0] is None or \
            trk.limits_ind[1] is None \
        or trk.saved[1]:
        return

    if len(trk.g_vars["seconds"]) == 0:
        return

    r_time = utils.get_seconds_from_fps(
        trk.g_vars["f_count"], trk.g_vars["fps"]
    )

    if int(r_time) <= trk.g_vars["seconds"][0][0]:
        return

    trk.saved[1] = True

    straight1 = trk.straights[trk.limits_ind[0]]
    angle1 = trk.angles[trk.limits_ind[0]]
    dim1 = trk.dimensions[trk.limits_ind[0]]

    if len(straight1) != 2:
        return

    straight_all = trk.straights[-1]
    angle_all = trk.angles[-1]
    dim_all = trk.dimensions[-1]


    # Calculate straight, angle and dimensions to the second fields.
    import math
    B = straight_all[1][0] - trk.x_limits[0]
    A = (B * math.degrees(math.sin(math.radians(angle_all[1])))) / math.degrees(math.sin(math.radians(angle_all[0])))
    y_r = int(straight_all[1][1] - A)

    straight2 = np.array([np.array([trk.x_limits[0], y_r]), straight_all[-1]])
    angle2 = np.copy(angle_all)
    dim2 = []
    dim2.append(straight2[1][0] - straight2[0][0])
    dim2.append(straight2[1][1] - straight2[0][1])
    dim2.append(math.sqrt(dim2[0]**2 + dim2[1]**2))


    if angle1[0] == 0 or angle1[1] == 0 or \
        angle2[0] == 0 or angle2[1] == 0:
        return

    area1 = trk.areas[trk.limits_ind[0]]
    area2 = trk.areas[-1]

    top, left, bottom, right = trk.boxes[trk.limits_ind[0]]
    perimeter1 = [right-left, bottom-top]

    top, left, bottom, right = trk.boxes[-1]
    perimeter2 = [right-left, bottom-top]

    counter = (len(trk.areas) - trk.limits_ind[0]) - 1
    time =  counter / trk.g_vars["fps"]

    # Set speed and lane in a variable
    secs = trk.g_vars["seconds"][0]
    speed_lane = ""
    if len(secs) >= 3:
        speed_lane = "{},{}".format(secs[1], secs[2])
    else:
        speed_lane = "{}".format(secs[1])

    # Angle1A,Angle1B,Angle2A,Angle2B,Cateto1A,Cateto1B,Hipotenusa1,Cateto2A,
    # Cateto2B,Hipotenusa2,Area1,Area2,Angle_all1,Angle_all2,Cateto_all1,
    # Cateto_all2,Hipotenusa_all,s1,s2,s3,s4,X_limit1,X_limit2,w1,h1,w2,
    # h2,fps,width,height,time,score,class,speed,lane(optional),frame counter
    line = "{},{},{},{},{},{},{},{},{},{},{},{},{:.15f},{:.15f},{:.15f},{:.15f},{:.15f},{},{},{}\n".format(
            ",".join(map(lambda x: "{:.15f}".format(x), angle1)),
            ",".join(map(lambda x: "{:.15f}".format(x), angle2)),
            ",".join(map(str, dim1)),
            ",".join(map(str, dim2)),
            area1,
            area2,
            ",".join(map(lambda x: "{:.15f}".format(x), angle_all)),
            ",".join(map(str, dim_all)),

            ",".join(map(lambda x: ",".join(map(str,x)), straight_all)),

            ",".join(map(str, trk.x_limits)),

            ",".join(map(str, perimeter1)),
            ",".join(map(str, perimeter2)),
            trk.g_vars["fps"],
            trk.g_vars["width"],
            trk.g_vars["height"],
            time,
            trk.score,
            trk.class_id,
            speed_lane,
            trk.g_vars["f_count"]
        )

    img_r = None
    img_path = None
    if img is not None:
        img_w = np.copy(img)
        utils.draw_all_boxes(img_w, [trk.box], box_color=(255,255,255))

        im3 = np.copy(img_w)
        im4 = np.copy(img_w)

        utils.draw_triangule(straight2, img_w)
        utils.draw_triangule(straight_all, im3)

        trk.imgs.append(img_w)
        trk.imgs.append(np.copy(im3))
        trk.imgs.append(np.copy(im4))

        img_r = np.concatenate((trk.imgs[:2]), axis=1)

        img_path = "{}/{}.jpg".format(trk.g_vars["root_path"], trk.g_vars["f_count"])

    del trk.g_vars["seconds"][0]

    if save_csv:
        cv2.imwrite(img_path, img_r)
        file_name = "{}/{}.csv".format(
            trk.g_vars["root_path"],
            trk.g_vars["file_name"]
        )
        with open(file_name, "a+") as f:
            f.write(line)

    return (line, img_r, img_path)

def detection(video_path, yolo, x_limits, cut_img=None, video_out=None,
    show=True, out_csv=False, set_time=False):
    max_age = 4
    min_hits = 1
    tracker_list =[]
    track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K'])
    vid = cv2.VideoCapture(video_path)

    ret, img = vid.read()
    fps = vid.get(cv2.CAP_PROP_FPS)

    if video_out is not None:
        width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS: {}, Width: {}, Height: {}".format(fps, width, height))
        out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (width, height)
            )

    left, right, top, bottom = cut_img[0], cut_img[1], cut_img[2], cut_img[3]

    if set_time:
        import time
        start_time = time.time()

    root_path = video_path.split("/")
    name = root_path[-1].split(".")[0]
    root_path = "/".join(root_path[:-1])

    seconds_file = None
    if out_csv:
        seconds_file = utils.csv_to_list("{}/{}.txt".format(root_path, name))

    g_vars = {
        "fps": fps,
        "f_count": 1,
        "seconds": seconds_file,
        "file_name": name,
        "root_path": root_path,
        "width": int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }

    while ret:
        if seconds_file is not None and len(seconds_file) == 0:
            vid.release()
            cv2.destroyAllWindows()
            break
        img = detector(img, yolo, max_age, min_hits,
            tracker_list, track_id_list, x_limits, g_vars,
            cut_img, out_csv)

        if set_time:
            seconds = time.time() - start_time
            utils.set_time_str(img, seconds)
            seconds = utils.get_seconds_from_fps(g_vars["f_count"], g_vars["fps"])
            utils.set_time_str(img, seconds, pos=(30, 60))
            utils.set_text(img, "FPS: {:.3f}, Current Frame: {}"
                .format(g_vars["fps"], g_vars["f_count"]), pos=(30, 90)
            )

        if x_limits is not None:
            cv2.line(img,(x_limits[0], 0),(x_limits[0], img.shape[0]),
                (255,0,0),1)

            cv2.line(img,(x_limits[1], 0),(x_limits[1], img.shape[0]),
                (255,0,0),1)

        if cut_img is not None:
            cv2.rectangle(img, (top, left),(bottom, right), (0, 0, 255), 1)

        if video_out is not None:
            out.write(img)

        if show:
            cv2.imshow('Testing', img)
        ret, img = vid.read()
        if show and (cv2.waitKey(25) & 0xFF == ord("q")):
            vid.release()
            cv2.destroyAllWindows()
            break
        g_vars["f_count"] += 1

    if video_out is not None:
        out.release()
    cv2.destroyAllWindows()
