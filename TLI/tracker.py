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
def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection
    and the union of two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''

    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) -
        np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) -
        np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])

    c = float(s_intsec)
    d = (s_a + s_b -s_intsec)
    if d == 0:
        return 0
    return c/d

class Tracker(): # class for Kalman Filter-based tracker
    def __init__(self):
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
        self.angle = []
        self.angles = []

        self.class_id = None
        self.score = None

    def update_R(self):
        R_diag_array = self.R_scaler * np.array([self.L, self.L,
            self.L, self.L])
        self.R = np.diag(R_diag_array)

    def add_box(self, box):
        self.box = box

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

        self.straight, self.angle = self.__gen_straight()
        self.straights.append(self.straight)
        self.angles.append(self.angle)

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
    def __gen_straight(self):
        track = np.array(self.track)
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
        straight.append(y_line)
        straight = np.array(straight).T

        A = 0.0
        B = 0.0
        import math

        if len(x) > 1:
            pt1 = straight[0]
            pt2 = straight[-1]

            A = math.atan((pt1[0] - pt2[0])/(pt1[1] - pt2[1] + 0.0000001))
            A = math.degrees(A)
            straight = [pt1, pt2]
            B = 90.0 - A

        return np.array(straight), [A, B]

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
            IOU_mat[t,d] = box_iou2(trk, det)

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
def detector(img, yolo, max_age, min_hits, tracker_list, track_id_list,
    cut_img=None, out_csv=None):

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
            tmp_trk = Tracker() # Create a new tracker
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
             if out_csv is not None:
                save_trk(trk, out_csv)
             img = utils.draw_tracker(img, trk)
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)

    for trk in deleted_tracks:
            track_id_list.append(trk.id)

    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    return img

def save_trk(trk, file_name):
    with open(file_name, "a+") as f:
        f.write("{},{},{},{}\n"
            .format(",".join(map(str, trk.angle)), trk.roc, trk.class_id,
                trk.score
            )
        )

def detection(video_path, yolo, cut_img=None, video_out=None, show=True,
    out_csv=None):
    max_age = 4
    min_hits = 1
    tracker_list =[]
    track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K'])
    vid = cv2.VideoCapture(video_path)

    ret, img = vid.read()

    if video_out is not None:
        fps = vid.get(cv2.CAP_PROP_FPS)
        width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS: {}, Width: {}, Height: {}".format(fps, width, height))
        out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (width, height)
            )

    while ret:
        img = detector(img, yolo, max_age,
            min_hits, tracker_list, track_id_list, cut_img, out_csv)

        if video_out is not None:
            out.write(img)

        if show:
            cv2.imshow('Testing', img)
        ret, img = vid.read()
        if show and (cv2.waitKey(25) & 0xFF == ord("q")):
            vid.release()
            cv2.destroyAllWindows()
            break

    if video_out is not None:
        out.release()
    cv2.destroyAllWindows()
