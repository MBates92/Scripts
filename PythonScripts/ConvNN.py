import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle

NAME = '100px-Unexponentiated-fbm-CNN'

pickle_in = open('Training/X.pickle','rb')
X = pickle.load(pickle_in)

pickle_in = open('Training/y.pickle','rb')
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='mse', optimizer = 'sgd', metrics=['accuracy'])

model.fit(X,y, batch_size=32, epochs=3, validation_split = 0.3, callbacks = [tensorboard])
model.save('100px-unexponentiated-fbm-CNN-{}.model'.format(int(time.time())))