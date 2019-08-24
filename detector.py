# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os

DARKNET = '/home/arrow/darknet/'
TF_POSE = '/home/arrow/tf-pose-estimation'
sys.path.insert(0, DARKNET)
sys.path.insert(0, TF_POSE)
sys.path.insert(0, DARKNET + 'data/')

import darknet as dn
import pdb
import cv2 
import threading
import argparse

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

HELMET = {
        'cfg': DARKNET + 'helmet/yolov3-tiny.cfg',
        'weights': DARKNET + 'helmet/yolov3-tiny_final.weights',
        'data': DARKNET + 'helmet/hardhat.data'}
OBJECT = {
        'cfg': DARKNET + 'cfg/yolov3-tiny.cfg',
        'weights': DARKNET + 'yolov3-tiny.weights',
        'data': DARKNET + 'cfg/coco.data'}

class Camera:
    def __init__(self):
        self.id = 0
        self.src = 0
        try:
            self.cap = cv2.VideoCapture(self.src)
        except:
            raise Exception("Failed to bring up device {}".format(self.src))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.cap.read()
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.thread_running = False

        self.width = self.cap.get(3)
        self.height = self.cap.get(4)

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.thread_running:
            raise Exception('Camera Thread is already running')
        self.thread_running = True
        print("[LOGS] STARTING CAMERA THREAD")
        self.thread = threading.Thread(target=self.grab_img, args=())
        self.thread.start()
        return self

    def grab_img(self):
        while self.thread_running:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            grabbed = self.grabbed
        return grabbed, frame_rgb

    def stop(self):
        self.thread_running = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


class Darknet:
    
    def __init__(self,args):
        dn.set_gpu(0)
        if args.object: 
            self.conf = .25
            model = OBJECT
        else:
            self.conf = .8
            model = HELMET
        self.net = dn.load_net(model['cfg'].encode("ascii"), 
                            model['weights'].encode("ascii"), 0, 1)
        self.meta = dn.load_meta(model['data'].encode("ascii"))
        self.dark_image = dn.make_image(dn.network_width(self.net),
                                        dn.network_height(self.net),
                                        3) 
    def detect(self):
        result = dn.detect_image(self.net, self.meta, self.dark_image, thresh=self.conf)
        return result 

class tfPose: 

    def __init__(self): 
        self.w, self.h = model_wh('432x368')
        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), 
                                target_size=(self.w, self.h), 
                                trt_bool=True)
        else:
            self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), 
                                target_size=(432, 368), 
                                trt_bool=True)
    def detect(self,frame):
        """
        Overlays pose of human on frame
        """
        humans = self.e.inference(frame,
                                resize_to_default=(self.w>0 and self.h>0),
                                upsample_size=4.0)
        print(humans)
        img = TfPoseEstimator.draw_humans(frame, 
                                        humans, 
                                        imgcopy=False)
        return img

def convertBack(x, y, w, h):
    xmin = int(round(x + (w )))
    xmax = int(round(x + (w*3)))
    if args.object: 
        xmin = int(round(x - (w /2)))
        xmax = int(round(x + (w /2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [0, 255, 0], 2)
    return img

def main():
    
    net = Darknet(args)
    if not args.object:
        pose = tfPose()
    cam = Camera()
    
    cam.start()
    
    count = 0
    while True: 
        ret ,frame = cam.read()
        if ret: 
            try:
                frame_resized = cv2.resize(frame, 
                                            (dn.network_width(net.net), 
                                            dn.network_height(net.net)),
                                            interpolation=cv2.INTER_LINEAR)
                dn.copy_image_from_bytes(net.dark_image, frame_resized.tobytes())
            except: 
                raise Exception("couldn't save current frame")
            res = net.detect()
            if res is not None and not args.object: 
                pose.detect(frame)
            img = cvDrawBoxes(res, frame_resized)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("demo", img)
            cv2.waitKey(1)
    cam.stop()
    cv2.destroyAllWindows()

if __name__=="__main__": 
    parser = argparse.ArgumentParser(description='Demo Detector written by Alexis Baudron')
    parser.add_argument('--object', type=bool, default=False,
                        help='Set to True for Yolov3 Object Detection')
    parser.add_argument('--helmet', type=bool, default=False,
                        help='Set to True for Helmet detection and Pose Estimation')
    parser.add_argument('--pose', type=bool, default=False,
                        help='Set tp True for pose estimation only')
    args = parser.parse_args()
    main()
