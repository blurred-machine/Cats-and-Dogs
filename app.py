# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:22:15 2020

@author: paras
"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib 

from flask import Flask, jsonify, request
import json
import flask

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


from io import BytesIO
import urllib


def loaderImage(URL):
    with urllib.request.urlopen(URL) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(125, 125))

    return image.img_to_array(img)


app = Flask(__name__)

IMAGE_SIZE = 64

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    print(form_data)
    print("Uploaded image path: "+str(form_data["image_path"]))

    #test_image = image.load_img(form_data["image_path"], target_size = (IMAGE_SIZE, IMAGE_SIZE))
    #test_image = image.img_to_array(test_image)
    
    test_image = loaderImage(form_data["image_path"][5:])
    
    test_image = np.expand_dims(test_image, axis = 0)
    print(test_image)
    

    classifier = joblib.load('prediction_classifier.pkl')
    result = classifier.predict(test_image)

    if result[0][0] == 1:
        prediction = 'Dog'
    else:
        prediction = 'Cat'    
    return flask.render_template('index.html', predicted_value="Uploaded image was of:\n {}".format(prediction))





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
