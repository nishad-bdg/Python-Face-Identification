# -*- coding: utf-8 -*-
import keras
import cv2
import os
import matplotlib.pyplot as plt

players = ['Ronaldo','Neymar','Messi']

def prepare(filepath):
	IMG_SIZE = 50  # 50 in txt-based
	img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
	return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# file path 
filepath = 'img1.jpg'

def predict_player(filepath):
    file = prepare(filepath)
    model = keras.models.load_model('CNN_MODEL')
    pred = model.predict_classes(file)
	# print(model.predict_proba(file))
    return players[pred[0]]
print(predict_player(filepath))
# processing for image detection
        
    
    

   
    
    

 






