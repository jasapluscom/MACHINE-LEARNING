#!/usr/bin/env python3
import os
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from PIL import Image, ImageDraw

class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
     
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
rcnn.load_weights('models/mask_rcnn_coco.h5', by_name=True)

arr = os.listdir("data")
for file in arr:
    img = load_img('data/' + file)
    img = img_to_array(img)
    results = rcnn.detect([img], verbose=0)
    r = results[0]
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for roi_list in r['rois']:
        cor = (roi_list[3],roi_list[2], roi_list[1], roi_list[0])
        draw.rectangle(cor, outline="red")
    img.show()
