import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D

DATA_PATH = './data/'
IMG_PATH = DATA_PATH + 'IMG/'

def getLinesfromCSV(csv_path):

    lines = []

    with open(csv_path) as csvfile:

        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:

            lines.append(line)

    return lines

def getImagePaths(path):

    lines = getLinesfromCSV(path)

    center_camera, left_camera, right_camera, measurements = [], [], [], []

    image_paths, steering_angles = [], []

    for line in lines:

        center_path, left_path, right_path = getFileNames(line)
        center_camera.append(center_path)
        left_camera.append(left_path)
        right_camera.append(right_path)

        measurements.append(float(line[3]))

    image_paths.extend(center_camera)
    image_paths.extend(left_camera)
    image_paths.extend(right_camera)

    steering_angles.extend(measurements)

    for m in measurements:
        steering_angles.append(m + 0.25)

    for m in measurements:
        steering_angles.append(m - 0.25)

    samples = list(zip(image_paths, steering_angles))

    return samples


def getFileNames(line):

    center_path, left_path, right_path = line[0], line[1], line[2]

    center_path = IMG_PATH + center_path.split('\\')[-1].strip()
    left_path = IMG_PATH + left_path.split('\\')[-1].strip()
    right_path = IMG_PATH + right_path.split('\\')[-1].strip()

    return center_path, left_path, right_path

def resize_images(image):
        return tf.image.resize_images(image, (80, 320))

def preProcess():
    
    model = Sequential()
    model.add(Lambda(resize_images, input_shape=(80,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    return model

def addNoise(img):
    h,w,c = img.shape

    noise = np.random.randint(0,50,(h, w)) # jitter/noise
    zitter = np.zeros_like(img)
    zitter[:,:,1] = noise

    noise_added = cv2.add(img, zitter)

    return noise_added


def reportSummary(model):
    with open('report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
