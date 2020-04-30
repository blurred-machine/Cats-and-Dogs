# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:21:11 2020

@author: paras
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


classifier = Sequential()
classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units= 128, activation="relu"))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])





