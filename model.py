# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:21:11 2020

@author: paras
"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator



# Images preprocessing before training
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')




classifier = Sequential()
classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, 3, 3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units= 128, activation="relu"))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 5, validation_data = test_set, validation_steps = 2000)
print(joblib.dump(classifier, 'prediction_classifier.pkl'))




# testing on a single input
test_image = image.load_img("./test_image_1.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(test_image)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat' 
print(prediction)
