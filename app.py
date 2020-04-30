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


app = Flask(__name__)



@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    print("Uploaded image path: "+str(form_data["input_image"]))

    test_image = image.load_img(form_data["input_image"], target_size = (32, 32))
    test_image = image.img_to_array(test_image)
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
