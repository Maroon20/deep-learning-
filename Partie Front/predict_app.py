# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:39:03 2019

@author: sanar
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from os import listdir
from os.path import isfile, join

from flask import jsonify
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# dimensions of our images
img_width, img_height = 150, 150
# categories
CATEGORIESG = ["Male", "Female"]

CATEGORIES = ["Apple", "Bat","Beetle"]
# load the model we saved
def get_model():
    global model
    model = load_model('model.h5')
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
    print(" * Models loaded!")
get_model()



# load the model we saved
def get_modelG():
    global modelG
    modelG = load_model('model-gender.h5')
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
    print(" * Models loaded!")
get_modelG()

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
#MPEG 7
@app.route('/predictResNet50', methods=['POST'])
def predict():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        img = image.load_img(file_path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=16)
    
    response = str(CATEGORIES[(classes[0])])

    return response

#MALE AND FEMALE
@app.route('/predictVGG16', methods=['POST'])
def predict1():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        img = image.load_img(file_path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = modelG.predict_classes(images, batch_size=16)
    
    response = str(CATEGORIESG[(classes[0][0])])

    return response

