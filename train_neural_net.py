import pickle
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split




pickle_in_X = open('X.pickle', 'rb')
X = pickle.load(pickle_in_X)

pickle_in_y = open('y.pickle','rb')
y = pickle.load(pickle_in_y) 

X = np.array(X).reshape(-1,50,50,1).astype('float32')
y = np.array(y)

X = X/255.0

one_hot_y = to_categorical(y)

INIT_LR = 0.01
batch_size = 64
EPOCHS = 41
num_classes = one_hot_y.shape[1:][0]


input_shape = (50,50,1)
chan_dim = 1

X_train,X_test,y_train,y_test = train_test_split(X,one_hot_y, test_size = 0.15, random_state = 42)


model = Sequential()
model.add(Conv2D(32, (3,3), padding = "same", input_shape = input_shape ))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = chan_dim))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))


#(CONV => RELU) *2 => POOL Layer set

model.add(Conv2D(64, (3,3), padding = "same"))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = chan_dim))
			
model.add(Conv2D(64, (3,3), padding = "same"))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = chan_dim))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = chan_dim))
model.add(Conv2D(128, (3,3), padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = chan_dim))
model.add(Conv2D(128, (3,3), padding = 'same'))
model.add(Activation ('relu'))
model.add(BatchNormalization(axis = chan_dim))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

# flatten image
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network
model.fit(X_train,y_train, batch_size = batch_size, epochs = EPOCHS, validation_data = (X_test,y_test))

model.save("NEW_MODEL")



