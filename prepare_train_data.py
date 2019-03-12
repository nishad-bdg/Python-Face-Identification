# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
import pickle



DATA_DIR = 'datasets/'
training_data = []
players_list = []

folders = os.listdir(DATA_DIR)
print(folders)

for item in folders[1:]:
    players_list.append(item)
    
def add_training_data():
    for player in players_list:
        print(player)
        new_path = "{}/".format(player)
        path = os.path.join(DATA_DIR,new_path)
        
        class_num = players_list.index(player)
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (50,50))
                training_data.append([new_array,class_num])
                print("Data appended")
            except:
                print("unable to append data")


add_training_data()

#random.shuffle(training_data)

X = []
y = []

    
for features,label in training_data:
    X.append(features)
    y.append(label)

#pickle save X data 
pickle_out = open("X.pickle", "wb")
pickle.dump(X,pickle_out)
pickle_out.close()

# pickle save y data
pickle_out = open("y.pickle", "wb")
pickle.dump(y,pickle_out)
pickle_out.close()
