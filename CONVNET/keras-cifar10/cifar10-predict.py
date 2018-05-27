#!/usr/bin/env python3
'''
predicting using cifar 10 trained data
'''
from keras.models import load_model
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys, os, getopt

def _load_model():
    try:
        model = load_model('cifar10convnet.h5')
        img_width, img_height = 32, 32
        model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    except:
        raise
    return model,img_width, img_height

def predict(path_img, model,img_width, img_height, np, image):
    try:
        kamus = {0:'airplane',1:'automobile',3:'bird',4:'cat', 5:'deer', 6:'dog', 7:'frog', 8:'horse', 9:'ship', 10:'truck'}
        img = image.load_img(path_img, target_size=(img_width, img_height))
        raw = np.array(img, dtype = float) / 255.0
        npimg = raw.reshape([-1, 3, img_width, img_height])
        classes = model.predict_classes(npimg)
        print("Predicted=", kamus[classes[0]])

    except:
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python3 {} <full image path to predict>". format(sys.argv[0]) )
        sys.exit()

    model,img_width, img_height = _load_model()
    path_img = sys.argv[1].strip()
    predict(path_img, model,img_width, img_height, np, image)
