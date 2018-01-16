import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.layers import Dropout, Flatten, Dense, Lambda, Conv2D, Cropping2D, SpatialDropout2D
from keras.optimizers import Adam

import sklearn
from sklearn.model_selection import train_test_split

import scipy.misc

DATA_PATH = './data/'
LOG_PATH = DATA_PATH + 'driving_log.csv'
IMG_PATH = DATA_PATH + 'IMG/'

MODEL_H5 = 'model.h5'

EPOCHS = 3
BATCH_SIZE = 64

from lib import getLinesfromCSV
from lib import getImagePaths
from lib import getFileNames
from lib import preProcess
from lib import reportSummary
from lib import addNoise

def images_generator(training_data, batch_size):

    data_size = len(training_data)

    while True:

        # Shuffle data set before splitting train and validation
        training_data = sklearn.utils.shuffle(training_data)

        for batch in range(0, data_size, batch_size):

            batch_data = training_data[ batch : batch + batch_size ]

            images = np.empty([3*batch_size, 80, 320, 3])
            steering_angles = np.empty([3*batch_size, 1])

            # Iterate over image paths and convert BGR to RGB
            for i, values in enumerate(batch_data):

                image_path = values[0]
                measurement = values[1]

                image = cv2.imread(image_path)
                image = image[55:135, : ]

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Add image and augmented flipped image
                noise_image = addNoise(image)
                flip_image = np.fliplr(image)

                images[i] = image
                images[len(batch_data)+i] = flip_image
                images[2*len(batch_data)+i] = noise_image
                steering_angles[i] = measurement
                steering_angles[len(batch_data)+i] = -measurement
                steering_angles[2*len(batch_data)+i] = measurement

            yield images, steering_angles

# NVidia CNN model with ELU activations
def model():
    model = preProcess()
    model.add(Conv2D(24,(5, 5), strides=(2,2), activation='elu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(36,(5, 5), strides=(2,2), activation='elu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(48,(5, 5), strides=(2,2), activation='elu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(SpatialDropout2D(0.2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    return model

training_data = getImagePaths(LOG_PATH)

# Split data set in 80-20 training-validation
training_data, validation_data = train_test_split(training_data, test_size=0.2)

samples_per_epoch = len(training_data)
nb_val_samples = len(validation_data)

print('Train Samples: {}'.format( len(training_data)))

train_generator = images_generator(training_data, BATCH_SIZE)
validation_generator = images_generator(validation_data, BATCH_SIZE)

model = model()

reportSummary(model)

# Compiling: Mean Square Error and Adam optimizer
model.compile(loss='mse', optimizer='adam')

#Training
model.fit_generator(
    train_generator,
    steps_per_epoch=samples_per_epoch,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=nb_val_samples,
    epochs=EPOCHS
)

model.save(MODEL_H5)
