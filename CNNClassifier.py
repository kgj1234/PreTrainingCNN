# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:49:59 2018
Keras Autoencoder: testing constrained interior constraint
@author: kgj1234
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from sklearn.preprocessing import OneHotEncoder



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.reshape(x_train,[60000,28,28,1])
x_test=np.reshape(x_test,[10000,28,28,1])
Enc=OneHotEncoder(n_values=10)
y_train=Enc.fit_transform(y_train.reshape((-1,1)))
print(y_train.shape)
inputs=Input(shape=(28,28,1))
x=Conv2D(32,(3,3),padding='same',activation='relu')(inputs)
x=Conv2D(32,(3,3),padding='same',activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(.25)(x)
x=Conv2D(64,(3,3),activation='relu')(x)
x=Conv2D(64,(3,3),activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(.25)(x)
x=Flatten()(x)
x=Dense(256,activation='relu')(x)
x=Dropout(.5)(x)
predictions=Dense(10,activation='softmax')(x)
model=Model(inputs=inputs,outputs=predictions)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_split=0.25, epochs=3, batch_size=16, verbose=1)
