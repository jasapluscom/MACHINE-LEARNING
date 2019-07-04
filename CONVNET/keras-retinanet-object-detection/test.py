#!/usr/bin/env python3
'''
cardboard detection test code using keras retinanet
modified from original example (jupyter notebook)
'''
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())
model = models.load_model("/home/ringlayer/Desktop/app/retinanet1/models/cb20.h5", backbone_name='resnet50')
model = models.convert_model(model)
print(model.summary())
labels_to_names = {0: 'cardboard'}

img_path = input("Image path : ")
image = read_image_bgr(img_path.strip())
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
image = preprocess_image(image)
image, scale = resize_image(image)
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)
boxes /= scale
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    if score < 0.5:
        break
    color = label_color(label)
    b = box.astype(int)
    draw_box(draw, b, color=color)
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
cv2.imshow('framename', draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
