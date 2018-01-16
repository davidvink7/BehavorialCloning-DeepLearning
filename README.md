# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/model.png "Model Visualization"
[image2]: ./output_images/BGR.jpg "Original Image"
[image3]: ./output_images/RGB.jpg "RGB Image"
[image4]: ./output_images/Flipped2.jpg "Flipped Image"
[image5]: ./output_images/noise_crop.jpg "Cropped Image with Noise"
[image6]: ./output_images/report.png "Summary report of CNN model"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* lib.py module with preprocessing functions to assist model.py

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

The model's first layer performs image normalization and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture and to be accelerated via GPU processing.

The network uses strided convolutions in the first three convolutional layers with a 2x2 stride and 5x5 kernel and a non-strided convolution with a 3x3 kernel size in the last two convolutional layers. The five convolutional layers are followed by three fully connected layers designed to return the steering angles.

The model includes ELU layers to introduce nonlinearity and takes care of the vanishing gradient problem with RELU. The data is normalized in the model using a Keras lambda layer and cropped to the effective area. Spatial dropout layers between the activations and dropouts before and after the fully connected layers reduce overfitting.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 111).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to minimize loss and overfitting during minimal processing time. 3 epochs and batch sizes of 64 samples were sufficient to achieve the appropiate result.

The model architecture is based on an end-to-end CNN [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by NVIDIA 

In order to gauge how well the model was working, I split the image and steering angle data into a training and validation set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

  Model Visualization           |  Tensorflow Summary Report    |
:------------------------------:|:-----------------------------:|
![model visualization][image1]  |  ![summary report][image6]    |

#### 3. Preprocessing Data Set and Augmentation

Augmentation of the data was done by flipping the images and angles. As well as cropping the effective part below the horizon and adding noise.

After the collection and augmentation process, I had 26,214 samples of data points. I then preprocessed this data by normalizing the images. I finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

#### 4. Dataset Samples of Images

  Original BGR image           |  Converted to RGB             |  Flipped Image                | Cropped Image with Noise      |
:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
![original BGR image][image2]  |  ![converted to RGB][image3]  | ![flipped image][image4]      | ![noise_crop image][image5]   |

