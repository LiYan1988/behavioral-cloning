# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os 
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.utils import shuffle

#%%
correction = 0.2

log = pd.read_csv('data-all/driving_log.csv')

log_center = log[['center', 'steering']]
log_center.columns = ['image', 'steering']

log_left = log[['left', 'steering']]
log_left.columns = ['image', 'steering']
log_left['steering'] = log_left['steering'] + correction

log_right = log[['right', 'steering']]
log_right.columns = ['image', 'steering']
log_right['steering'] = log_right['steering'] - correction


log = pd.concat([log_center, log_left, log_right], ignore_index=True)
log = log.applymap(lambda x: x.strip() if isinstance(x, str) else x)
log.to_csv('log_all_cameras.csv', index=False)

#%%
def read_image(image_path):
    img = cv2.imread(image_path)
    img = img[...,::-1]
    return img

def data_generator(log, batch_size=32):
    n_samples = log.shape[0]
    
    while 1:
        log_shuffled = log.sample(frac=1)
        
        for offset in range(0, n_samples, batch_size):
            n_flip = np.random.randint(batch_size)
            batch_samples = log_shuffled.iloc[offset:offset+batch_size]
            batch_size_now = batch_samples.shape[0]
            
            images = []
            angles = []
            weights = []
            for n in range(batch_size_now):
                img = read_image(os.path.join('data-all', batch_samples.iloc[n, 0]))
                if n<n_flip:
                    images.append(img[:,::-1,:])
                    angles.append(-batch_samples.iloc[n, 1])
                else:
                    images.append(img)
                    angles.append(batch_samples.iloc[n, 1])
                weights.append(len(os.path.split(batch_samples.iloc[n, 0])))
                    
            x_train = np.array(images)
            y_train = np.array(angles)
            x_train, y_train = shuffle(x_train, y_train)
            
            yield tuple(x_train, y_train, weights)
                    
#%%
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.core import Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
#print('Normol:', model.output.get_shape().as_list())
model.add(Convolution2D(24, 5, 5, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))
#print('Conv1:', model.output.get_shape().as_list())
model.add(Convolution2D(36, 5, 5, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))
#print('Conv2:', model.output.get_shape().as_list())
model.add(Convolution2D(48, 5, 5, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))
#print('Conv3:', model.output.get_shape().as_list())
model.add(Convolution2D(64, 2, 2, activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))
#print('Conv4:', model.output.get_shape().as_list())
model.add(Convolution2D(64, 3, 3, activation='relu'))
#print('Conv5:', model.output.get_shape().as_list())
model.add(Flatten())
#print('Flatten:', model.output_shape)
model.add(Dense(1164))
#print('FC1:', model.output_shape)
model.add(Dense(100))
#print('FC2:', model.output_shape)
model.add(Dense(50))
#print('FC3:', model.output_shape)
model.add(Dense(10))
#print('FC4:', model.output_shape)
model.add(Dense(1))
#print('FC5:', model.output_shape)

log_shuffled = log.sample(frac=1)
size_total = log_shuffled.shape[0]
size_train = int(size_total * 0.7)
size_test = int(size_total * 0.1)
size_validation = size_total - size_train - size_test

log_train = log_shuffled.iloc[:size_train]
log_test = log_shuffled.iloc[size_train:size_train+size_test]
log_validation = log_shuffled.iloc[size_train+size_test:]

generator_train = data_generator(log_train, 32)
generator_test = data_generator(log_test, 32)
generator_validation = data_generator(log_validation, 32)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(generator_train, 
                    samples_per_epoch=size_train,
                    nb_epoch=5,
                    validation_data=generator_validation,
                    nb_val_samples=size_validation)

model.evaluate_generator(generator_test,  val_samples=size_test)
model.save('model.h5')