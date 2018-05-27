#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 09:47:14 2018

@author: Antonius Ringlayer
www.ringlayer.net - www.ringlayer.com
@ringlayer

predict image using vgg16 pretrained model
without mask rcnn
"""

from gtts import gTTS
import os, sys
import numpy as np
from resnet50 import ResNet50
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

class recognizer(): 
    def tts_online_gtts_speak(self, words, speed='slow'):
        try:
            is_slow = True
            if (speed == "fast"):
                is_slow = False
            tts = gTTS(text= words, lang='en', slow=is_slow).save("/tmp/tmp.mp3")
            os.system("killall -9 gnome-mplayer;killall -9 mplayer")
            cmd = "gnome-mplayer --volume 150 -q /tmp/tmp.mp3"
            pipe = os.popen(cmd).read()
            print (pipe)
        except:
           raise    
           
           
    def predictor(self, sys):
        try:
            if len(sys.argv) < 2:
                print("Usage : ./saying-object.py <image path>")
                sys.exit()
                
            img_path = sys.argv[1]
             
            model = VGG16(weights="imagenet")
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            objek = "?"
            preds = model.predict(x)
            pred_dict = decode_predictions(preds)
            objek = pred_dict[0][0][1]
            if len(objek) > 1:
                words = "Your object detected as " + objek.replace("_"," ") 
                self.tts_online_gtts_speak(words)
            else:
                words = "sorry sir failed to predict your image"
                self.tts_online_gtts_speak(words)
        except:
            raise


ring = recognizer()
ring.predictor(sys)
