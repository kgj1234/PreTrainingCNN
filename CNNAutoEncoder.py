# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:44:43 2018

@author: kgj1234
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:49:59 2018
Keras Autoencoder: testing constrained interior constraint
@author: kgj1234
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from sklearn.preprocessing import OneHotEncoder



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.reshape(x_train,[60000,28,28,1])
x_test=np.reshape(x_test,[10000,28,28,1])
Enc=OneHotEncoder(n_values=10)
y_train=Enc.fit_transform(y_train.reshape((-1,1)))

inputs=Input(shape=(28,28,1))
x=Conv2D(32,(3,3),padding='same',activation='relu')(inputs)
x=Conv2D(32,(3,3),padding='same',activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(.25)(x)
x=Conv2D(64,(3,3),padding='same',activation='relu')(x)

x=Conv2D(64,(3,3),padding='same',activation='relu')(x)
x=MaxPooling2D(pool_size=(2,2))(x)

x=Dropout(.25)(x)

flatten=Flatten()(x)

x=Dense(256,activation='relu')(flatten)
x=Dropout(.5)(x)
predictions=Dense(10,activation='softmax')(x)
x=Dropout(.5)(predictions)
x=Dense(256,activation='relu')(x)
x=Dense((7*7*64))(x)
x=Reshape((7,7,64),input_shape=(7*7*64,))(x)

x=UpSampling2D(size=(2,2))(x)

x=Conv2D(64,(3,3),padding='same',activation='relu')(x)
x=Conv2D(64,(3,3),padding='same',activation='relu')(x)
x=UpSampling2D(size=(2,2))(x)
x=Conv2D(32,(3,3),padding='same',activation='relu')(x)
x=Conv2D(32,(3,3),padding='same',activation='relu')(x)
x=Flatten()(x)
final=Dense(28*28)(x)

final=Reshape((28,28,1))(final)




model=Model(inputs=inputs,outputs=[predictions,final])

model.compile(optimizer='rmsprop',loss=['categorical_crossentropy','kullback_leibler_divergence'],loss_weights=[2,.5],metrics=['accuracy','mean_squared_error'])

history = model.fit(x_train,[y_train,x_train], validation_split=0.25, epochs=200, batch_size=16, verbose=1)
