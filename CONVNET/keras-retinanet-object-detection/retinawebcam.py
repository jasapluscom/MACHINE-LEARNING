#!/usr/bin/env python3
'''
cardboard detection from webcam image using keras retinanet
created by Antonius Ringlayer
'''
import cv2, os, time, keras
import numpy as np
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class Cardboard:
    obj_num = 0
    x = 0
    y = 0
    def __init__(self, num, x, y, b):
        self.obj_num = num
        self.x = x
        self.y = y
        self.b = b

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

os.system("echo '' > logs/webcam.txt")
f = open("logs/webcam.txt", "a")
keras.backend.tensorflow_backend.set_session(get_session())
#model = models.load_model("/home/ringlayer/Desktop/app/retinanet1/models/resnet50_09.h5", backbone_name='resnet50')
model = models.load_model("/home/ringlayer/Desktop/app/retinanet1/models/cb20.h5", backbone_name='resnet50')
model = models.convert_model(model)
print(model.summary())
labels_to_names = {0: 'cardboard'}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    try:
        current_cardboard = 0
        box_lists = []
        ret, image = cap.read()
        draw = image.copy()
        draw = cv2.resize(draw,(640,480))
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        start = time.time()
        boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))

        boxes /= scale

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.5:
                break
            current_cardboard += 1
            color = label_color(label)
            b = box.astype(int)
            x = b[0]
            y = b[1]
            box_lists.append(Cardboard(current_cardboard, x, y, b))
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

        draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        cv2.imshow('framename', draw)
        if current_cardboard > 0:
            os.system("clear")
            str_data = "\n\n====last cardboard data===="
            for obj in box_lists:
                str_data += "\ngot cardboard : " + str(obj.obj_num)
                str_data += "\ngot x : " + str(obj.x)
                str_data += "\ngot y : " + str(obj.y)
                str_data += "\nfull coordinate : " + str(obj.b)
                str_data += "\n"
            f.write(str_data)
            print(str_data)
    except:
	    pass

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()

f.close()
