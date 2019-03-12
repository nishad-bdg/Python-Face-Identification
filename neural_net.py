# -*- coding: utf-8 -*-
import pickle
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU


pickle_in_X = open('X.pickle', 'rb')
X = pickle.load(pickle_in_X)

pickle_in_y = open('y.pickle','rb')
y = pickle.load(pickle_in_y) 

X = np.array(X).reshape(-1,50,50,1).astype('float32')
y = np.array(y)

X = X/255.0

one_hot_y = to_categorical(y)


batch_size = 32
epochs = 15
num_classes = one_hot_y.shape[1:][0]

# 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,one_hot_y, test_size = 0.1, random_state = 0)


model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), activation = 'linear', input_shape = (50,50,1), padding = 'same' ))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, kernel_size = (3,3), activation = 'linear',  padding = 'same' ))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, kernel_size = (3,3), activation = 'linear',  padding = 'same' ))
model.add(LeakyReLU(alpha = 0.1))
model.add(MaxPooling2D(2,2))

model.add(Dense(256, activation = 'linear'))
model.add(LeakyReLU(alpha = 0.1))

model.add(Flatten())
model.add(Dense(num_classes, activation = 'softmax'))

#compile the model 
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

# fit the model
model.fit(X_train,y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test,y_test))

# saving fitted model for future use
model.save('CNN_MODEL_LAST')

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score : {}'.format(score))
print('Test accuracy : {}'.format(acc))






