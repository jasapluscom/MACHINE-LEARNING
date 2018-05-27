#!/usr/bin/env python3
'''
Convnet model using CIFAR-10 dataset
based on

https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

trained on one of my laptop using GeForce 940MX
ringlayer@ringlayer-Inspiron-3442:~/Desktop/keras/convnet/convnet_object_recognition$ ./train-and-save-model.py
/usr/local/lib/python3.5/dist-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
2018-05-27 03:16:32.110305: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-05-27 03:16:32.244811: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-05-27 03:16:32.245524: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: GeForce 940MX major: 5 minor: 0 memoryClockRate(GHz): 0.8605
pciBusID: 0000:01:00.0
totalMemory: 1.96GiB freeMemory: 1.82GiB
2018-05-27 03:16:32.245551: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-05-27 03:16:34.056400: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-05-27 03:16:34.056438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-05-27 03:16:34.056450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-05-27 03:16:34.056650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1582 MB memory) -> physical GPU (device: 0, name: GeForce 940MX, pci bus id: 0000:01:00.0, compute capability: 5.0)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 32, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 32)        9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 16, 16)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 64, 16, 16)        18496
_________________________________________________________________
dropout_2 (Dropout)          (None, 64, 16, 16)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 64, 16, 16)        36928
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 64, 8, 8)          0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 128, 8, 8)         73856
_________________________________________________________________
dropout_3 (Dropout)          (None, 128, 8, 8)         0
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 128, 8, 8)         147584
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 128, 4, 4)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0
_________________________________________________________________
dropout_4 (Dropout)          (None, 2048)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              2098176
_________________________________________________________________
dropout_5 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_2 (Dense)              (None, 512)               524800
_________________________________________________________________
dropout_6 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130
=================================================================
Total params: 2,915,114
Trainable params: 2,915,114
Non-trainable params: 0
_________________________________________________________________
None
WARNING:tensorflow:Variable *= will be deprecated. Use variable.assign_mul if you want assignment to the variable value or 'x = x * y' if you want a new python Tensor object.
Train on 50000 samples, validate on 10000 samples
Epoch 1/25
50000/50000 [==============================] - 51s 1ms/step - loss: 1.9325 - acc: 0.2891 - val_loss: 1.7165 - val_acc: 0.3904
Epoch 2/25
50000/50000 [==============================] - 48s 966us/step - loss: 1.5208 - acc: 0.4458 - val_loss: 1.3991 - val_acc: 0.4998
Epoch 3/25
50000/50000 [==============================] - 48s 950us/step - loss: 1.3383 - acc: 0.5150 - val_loss: 1.3173 - val_acc: 0.5259
Epoch 4/25
50000/50000 [==============================] - 47s 937us/step - loss: 1.2092 - acc: 0.5661 - val_loss: 1.1313 - val_acc: 0.5955
Epoch 5/25
50000/50000 [==============================] - 47s 946us/step - loss: 1.0967 - acc: 0.6094 - val_loss: 1.1068 - val_acc: 0.6063
Epoch 6/25
50000/50000 [==============================] - 48s 954us/step - loss: 1.0070 - acc: 0.6399 - val_loss: 0.9394 - val_acc: 0.6707
Epoch 7/25
50000/50000 [==============================] - 47s 946us/step - loss: 0.9278 - acc: 0.6701 - val_loss: 0.8610 - val_acc: 0.6946
Epoch 8/25
50000/50000 [==============================] - 48s 956us/step - loss: 0.8665 - acc: 0.6928 - val_loss: 0.8392 - val_acc: 0.7080
Epoch 9/25
50000/50000 [==============================] - 48s 963us/step - loss: 0.8103 - acc: 0.7128 - val_loss: 0.8229 - val_acc: 0.7064
Epoch 10/25
50000/50000 [==============================] - 47s 934us/step - loss: 0.7626 - acc: 0.7300 - val_loss: 0.7570 - val_acc: 0.7387
Epoch 11/25
50000/50000 [==============================] - 48s 953us/step - loss: 0.7181 - acc: 0.7457 - val_loss: 0.7392 - val_acc: 0.7422
Epoch 12/25
50000/50000 [==============================] - 48s 962us/step - loss: 0.6836 - acc: 0.7591 - val_loss: 0.7001 - val_acc: 0.7582
Epoch 13/25
50000/50000 [==============================] - 48s 953us/step - loss: 0.6526 - acc: 0.7689 - val_loss: 0.6820 - val_acc: 0.7648
Epoch 14/25
50000/50000 [==============================] - 49s 974us/step - loss: 0.6226 - acc: 0.7794 - val_loss: 0.6755 - val_acc: 0.7670
Epoch 15/25
50000/50000 [==============================] - 47s 933us/step - loss: 0.5977 - acc: 0.7896 - val_loss: 0.6639 - val_acc: 0.7704
Epoch 16/25
50000/50000 [==============================] - 46s 914us/step - loss: 0.5790 - acc: 0.7946 - val_loss: 0.6509 - val_acc: 0.7757
Epoch 17/25
50000/50000 [==============================] - 49s 986us/step - loss: 0.5544 - acc: 0.8036 - val_loss: 0.6437 - val_acc: 0.7794
Epoch 18/25
50000/50000 [==============================] - 51s 1ms/step - loss: 0.5372 - acc: 0.8103 - val_loss: 0.6409 - val_acc: 0.7814
Epoch 19/25
50000/50000 [==============================] - 50s 997us/step - loss: 0.5189 - acc: 0.8174 - val_loss: 0.6259 - val_acc: 0.7860
Epoch 20/25
50000/50000 [==============================] - 50s 996us/step - loss: 0.5015 - acc: 0.8213 - val_loss: 0.6334 - val_acc: 0.7836
Epoch 21/25
50000/50000 [==============================] - 53s 1ms/step - loss: 0.4823 - acc: 0.8280 - val_loss: 0.6227 - val_acc: 0.7894
Epoch 22/25
50000/50000 [==============================] - 52s 1ms/step - loss: 0.4673 - acc: 0.8345 - val_loss: 0.6234 - val_acc: 0.7886
Epoch 23/25
50000/50000 [==============================] - 52s 1ms/step - loss: 0.4570 - acc: 0.8389 - val_loss: 0.6110 - val_acc: 0.7965
Epoch 24/25
50000/50000 [==============================] - 51s 1ms/step - loss: 0.4423 - acc: 0.8428 - val_loss: 0.6378 - val_acc: 0.7849
Epoch 25/25
50000/50000 [==============================] - 48s 951us/step - loss: 0.4315 - acc: 0.8465 - val_loss: 0.6076 - val_acc: 0.7957
Accuracy: 79.57%
'''
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

#train
numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

'''
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
print("Accuracy: %.2f%%" % (scores[1]*100))
'''

savethis = input("save model ? (y/n)")
if savethis == "y":
    model.save('cifar10convnet.h5')
    print("[+] model saved")
