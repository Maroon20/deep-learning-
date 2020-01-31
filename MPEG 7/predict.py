# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:39:03 2019

@author: sanar
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join


# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

mypath = "predict/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
# predicting images
apple_counter = 0 
bat_counter  = 0
beetle_counter  = 0
for file in onlyfiles:
    img = image.load_img(mypath+file, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=16)

    
    
    if classes == 0:
        print(file + ": " + 'apple')
        apple_counter += 1
    elif classes == 1:
        print(file + ": " + 'bat')
        bat_counter += 1
    else:
        print(file + ": " + 'beetle')
        beetle_counter += 1


print("Apples:",apple_counter)
print("Bats :",bat_counter)
print("Beetles :",beetle_counter)



