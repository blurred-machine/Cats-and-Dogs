# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:21:11 2020

@author: paras
"""

import pandas as pd
import numpy as np
import os
from sklearn.externals import joblib 

from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.preprocessing import image
from tensorflow.contrib.keras import backend


IMAGE_SIZE = 128
BATCH_SIZE = 32

# Images preprocessing before training
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("./dataset/training_set",
                                                 target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory("./dataset/test_set",
                                            target_size = (IMAGE_SIZE, IMAGE_SIZE),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'binary')





class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units= 64, activation="relu"))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 5, validation_data = test_set, validation_steps = 2000)

history = LossHistory()

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/BATCH_SIZE,
                         epochs=2,
                         validation_data=test_set,
                         validation_steps=2000/BATCH_SIZE,
                         workers=12,
                         callbacks=[history])


print(joblib.dump(classifier, 'prediction_classifier.pkl'))


# Save model
model_backup_path = "./cats_and_dogs_model.h5"
classifier.save(model_backup_path)
print("Model saved to: ", model_backup_path)
 
# Save loss history to file
loss_history_path = "./loss_history.log"
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()
 
backend.clear_session()
print("The model class indices are: ", training_set.class_indices)






# testing on a single input
test_image = image.load_img("./test_image_2.jpg", target_size = (IMAGE_SIZE, IMAGE_SIZE))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(test_image)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat' 
print(prediction)

import tensorflow
tensorflow.__version__