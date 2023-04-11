from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Conv2D ,Flatten,Dropout,MaxPool2D, BatchNormalization
from keras.utils import np_utils
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory  
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
import keras
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
import pandas as pd
import seaborn

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10, fill_mode='nearest',
    vertical_flip= True,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range = (0.4,0.6),
    horizontal_flip=True,validation_split=0.3)

train_ds = datagen.flow_from_directory(
    directory='./train',
    target_size=(60, 60),
    batch_size=32,
    class_mode='binary',subset='training'
)


validation_ds = datagen.flow_from_directory(
    directory='./train',
    target_size=(60, 60),
    batch_size=32,
    class_mode='binary',subset='validation'
)

vgg_model =  VGG19(include_top=True , weights='imagenet')
for models in vgg_model.layers:
  models.trainable= False

vgg_model = keras.Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)
model = keras.Sequential()
for layer in vgg_model.layers:
  model.add(layer)

model.add(Dense(4, activation='softmax'))
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
modelfit = model.fit(train_ds,
    validation_data = validation_ds, 
    callbacks = [early],
    epochs = 50)

print(modelfit)

'''
train_ds = datagen.image_dataset_from_directory(
    directory='./train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(60, 60),
    validation_split=0.2,
    seed=123,
    subset='training'
    )

validation_ds = image_dataset_from_directory(
    directory='./train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(60, 60),
    validation_split=0.2,
    seed=123,
    subset='validation'
    )
'''
print(train_ds)