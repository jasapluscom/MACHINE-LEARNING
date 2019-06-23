#!/usr/bin/env python3
'''
yolo_webcam.py
load yolo weight to detect object on webcam streaming
'''
import cv2
import numpy as np
from math import floor
from pydarknet import Detector, Image
import cv2
net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0, bytes("cfg/coco.data",encoding="utf-8"))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    img_darknet = Image(frame)
    results = net.detect(img_darknet)
    for cat, score, bounds in results:
        clas = (str(cat.decode("utf-8")))
        x, y, w, h = bounds
        print("x of middle:" , floor(x) , " px, y of middle:" , floor(y) , " px")
        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
        cv2.putText(frame,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()