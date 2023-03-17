from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import cv2
from google.colab.patches import cv2_imshow
import re
import datetime
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate 
from tensorflow.keras.layers import Input, Add, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import callbacks

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
# %matplotlib inline

from IPython.display import HTML
from base64 import b64encode

#Import required Image library

path1 = '/content/drive/MyDrive/KITISS_DATASET_LANEDETECTION/training/image_2/'   
path2 = '/content/drive/MyDrive/KITISS_DATASET_LANEDETECTION/training/vignette_images/'
path3 = '/content/drive/MyDrive/KITISS_DATASET_LANEDETECTION/testing/test_images/'
path4 = '/content/drive/MyDrive/night_drive_data_BUSAN/'


# Training Image Pre-Processing
listing = os.listdir(path1)  

for file in listing:
    
    input_image = cv2.imread(path1+file)

    #resizing the image according to our need
    # resize() function takes 2 parameters,
    # the image and the dimensions
    input_image = cv2.resize(input_image, (480, 480))

    # Extracting the height and width of an image
    rows, cols = input_image.shape[:2]

    # generating vignette mask using Gaussian
    # resultant_kernels
    X_resultant_kernel = cv2.getGaussianKernel(cols,200)
    Y_resultant_kernel = cv2.getGaussianKernel(rows,200)

    #generating resultant_kernel matrix
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

    #creating mask and normalising by using np.linalg
    # function
    mask = 75 * resultant_kernel / np.linalg.norm(resultant_kernel)
    output = np.copy(input_image)
    # mask1 = 200 * resultant_kernel / np.linalg.norm(resultant_kernel)
    # output = np.copy(input_image)

    # applying the mask to each channel in the input image
    for i in range(3):
      output[:,:,i] = output[:,:,i] * mask

  # define the contrast and brightness value
    contrast = 1 # Contrast control ( 0 to 127)
    brightness = 1 # Brightness control (0-100)

  # call addWeighted function. use beta = 0 to effectively only
    #operate on one image
    output = cv2.addWeighted( output, contrast, output, 0, brightness)
    output=cv2.resize(output,(1240,375))
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # cv2_imshow(output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    filename = path2+file
    filename = re.sub(r".jpg",".png",filename)
    cv2.imwrite(filename,output)


# Testing Image Pre-Processing
testing = os.listdir(path4)  
for file in testing:
    input_image = cv2.imread(path4+file)
    output = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    output=cv2.resize(output,(1240,375))
    output=cv2.resize(output,(1240,375))
    # cv2_imshow(output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    filename = path3+file
    filename = re.sub(r".jpg",".png",filename)
    cv2.imwrite(filename,output)

# Load directories
train_data_dir = "/content/drive/MyDrive/KITISS_DATASET_LANEDETECTION/training/image_2/"
train_gt_dir = "/content/drive/MyDrive/KITISS_DATASET_LANEDETECTION/training/gt_image_2/"

test_data_dir = "/content/drive/MyDrive/KITISS_DATASET_LANEDETECTION/testing/test_images/"

# Number of training examples
TRAINSET_SIZE = int(len(os.listdir(train_data_dir)) * 0.8)
print(f"Number of Training Examples: {TRAINSET_SIZE}")

VALIDSET_SIZE = int(len(os.listdir(train_data_dir)) * 0.1)
print(f"Number of Validation Examples: {VALIDSET_SIZE}")

TESTSET_SIZE = int(len(os.listdir(test_data_dir)))
print(f"Number of Testing Examples: {TESTSET_SIZE}")

# Initialize Constants
IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 1
SEED = 123

# Function to load image and return a dictionary
def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)       # decodes the image to uint8 tensor
    image = tf.image.convert_image_dtype(image, tf.uint8) # returns a converted image to dtype mentioned

    # Three types of img paths: um, umm, uu
    # gt image paths: um_road, umm_road, uu_road
    mask_path = tf.strings.regex_replace(img_path, "image_2", "gt_image_2")   #regex of tenserflow
    mask_path = tf.strings.regex_replace(mask_path, "um_", "um_road_")
    mask_path = tf.strings.regex_replace(mask_path, "umm_", "umm_road_")
    mask_path = tf.strings.regex_replace(mask_path, "uu_", "uu_road_")
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)     #decodes png encoded image to uint8 tensor (by default)
    
    non_road_label = np.array([255, 0, 0])
    road_label = np.array([255, 0, 255])
    other_road_label = np.array([0, 0, 0])
    
    # Convert to mask to binary mask
    mask = tf.experimental.numpy.all(mask == road_label, axis = 2)    #tensorflow version of numpy.all.... checks if elements evaluate to True
    mask = tf.cast(mask, tf.uint8)                                    #casts a tensor to new dtype
    mask = tf.expand_dims(mask, axis=-1)                              #returns a tensor with same data as input, additional dimension added...\
                                                                      #no change in case of axis = -1

    return {'image': image, 'segmentation_mask': mask}

# Generate dataset variables
all_dataset = tf.data.Dataset.list_files(train_data_dir + "*.png", seed=SEED)   #   A dataset of all files matching one or more patterns.
all_dataset = all_dataset.map(parse_image)                                      #   maps the function across the dataset

train_dataset = all_dataset.take(TRAINSET_SIZE + VALIDSET_SIZE)
val_dataset = train_dataset.skip(TRAINSET_SIZE)
train_dataset = train_dataset.take(TRAINSET_SIZE)
test_dataset = tf.data.Dataset.list_files(test_data_dir + "*.png", seed=SEED)
test_dataset = test_dataset.map(parse_image)
test_dataset = test_dataset.take(TESTSET_SIZE)

type(all_dataset)

# Tensorflow function to rescale images to [0, 1]
@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

# Tensorflow function to apply preprocessing transformations
@tf.function
def load_image_train(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)          #horizontally flips the image
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

# Tensorflow function to preprocess validation images
@tf.function
def load_image_test(datapoint: dict) -> tuple:                                          #   datapoint: dict  ==>    type hinting
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

BATCH_SIZE = 32
BUFFER_SIZE = 1000

dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

# -- Train Dataset --#
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)   # num_parallel_calls=tf.data.AUTOTUNE :
                                                                                                # no. of elements to process in parallel
                                                                                                # if set to tf.data.AUTOTUNE then no. of parallel calls set
                                                                                                #   dynamically based on available CPU

dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)           # shuffles the dataset > to generalise the training and reduce the variance
                                                                                          # avoid overfitting

dataset['train'] = dataset['train'].repeat()                                              # to train the data various times
dataset['train'] = dataset['train'].batch(BATCH_SIZE)                                     # batch size is number of samples processed before the model is updated

dataset['train'] = dataset['train'].prefetch(buffer_size=tf.data.AUTOTUNE)                # prefetch allows later elements to be prepared while current element is being processed
                                                                                          # buffer_size =  number of elements that will be fetched
                                                                                          # if set to tf.data.AUTOTUNE then no. of parallel calls set
                                                                                          #   dynamically based on available CPU

#-- Validation Dataset --#
dataset['val'] = dataset['val'].map(load_image_test)
dataset['val'] = dataset['val'].repeat()
dataset['val'] = dataset['val'].batch(BATCH_SIZE)
dataset['val'] = dataset['val'].prefetch(buffer_size=tf.data.AUTOTUNE)

#-- Testing Dataset --#
dataset['test'] = dataset['test'].map(load_image_test)
dataset['test'] = dataset['test'].batch(BATCH_SIZE)
dataset['test'] = dataset['test'].prefetch(buffer_size=tf.data.AUTOTUNE)

print(dataset['train'])
print(dataset['val'])
print(dataset['test'])

# Function to view the images from the directory
def display_sample(display_list):
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))   # Converts a 3D numpy array to a PIL image
        plt.axis('off')
        
    plt.show()
    
for image, mask in dataset["train"].take(1):
    sample_image, sample_mask = image, mask

display_sample([sample_image[0], sample_mask[0]])

# Get VGG-16 network as backbone
vgg16_model = VGG16()
vgg16_model.summary()

# Define input shape
input_shape = (IMG_SIZE, IMG_SIZE, N_CHANNELS)

# Generate a new model using the VGG network
# Input
inputs = Input(input_shape)

# VGG network
vgg16_model = VGG16(include_top = False, weights = 'imagenet', input_tensor = inputs)

# Encoder Layers
c1 = vgg16_model.get_layer("block3_pool").output         
c2 = vgg16_model.get_layer("block4_pool").output         
c3 = vgg16_model.get_layer("block5_pool").output         

# Decoder
u1 = UpSampling2D((2, 2), interpolation = 'bilinear')(c3)

d1 = Concatenate()([u1, c2])

u2 = UpSampling2D((2, 2), interpolation = 'bilinear')(d1)
d2 = Concatenate()([u2, c1])

# Output
u3 = UpSampling2D((8, 8), interpolation = 'bilinear')(d2)
outputs = Conv2D(N_CLASSES, 1, activation = 'sigmoid')(u3)

model = Model(inputs, outputs, name = "VGG_FCN8")

m_iou = tf.keras.metrics.MeanIoU(2)      #  MeanIoU : Intersection-Over-Union is a common evaluation metric for image segmentation
model.compile(optimizer=Adam(),
              loss=BinaryCrossentropy(),
              metrics=[m_iou])

# Function to create a mask out of network prediction
def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    # Round to closest
    pred_mask = tf.math.round(pred_mask)             # rounds the value of tensor to nearest integer, rounds half to even
    
    # [IMG_SIZE, IMG_SIZE] -> [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

# Function to show predictions
def show_predictions(dataset=None, num=1):
    if dataset:
        # Predict and show image from input dataset
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], true_mask, create_mask(pred_mask)])
    else:
        # Predict and show the sample image
        inference = model.predict(sample_image)
        display_sample([sample_image[0], sample_mask[0],
                        inference[0]])
        
for image, mask in dataset['train'].take(1):
    sample_image, sample_mask = image, mask

show_predictions()

# Callbacks and Logs
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))    # strftime = converts a datetime object containing current date and time to different string formats.

callbacks = [
    DisplayCallback(),
    callbacks.TensorBoard(logdir, histogram_freq = -1),                             # visualization tool available for Tensorflow
    callbacks.EarlyStopping(patience = 10, verbose = 1),                            # used to stop training when a monitored metric has stopped improving...
                                                                                    # patience = 10  :  no. of epochs with no improvement after which training will be stopped
                                                                                    # verbose = 1   : mode = 0 silent, mode = 1 displays messages when callback takes an action

    callbacks.ModelCheckpoint('best_model.h5', verbose = 1, save_best_only = True)  # callback to save the keras model or model weights at some frequency
                                                                                    # filepath = path + filename
                                                                                    # save_best_only = by default False
                                                                                    #      if True it only saves the model and weights when the model is considered the best and latest model
                                                                                    #  according the metric monitored...
]
        
# Set Variables
EPOCHS = 500
STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALIDSET_SIZE // BATCH_SIZE

model_history = model.fit(dataset['train'], epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data = dataset["val"],
                          validation_steps=VALIDATION_STEPS,
                          callbacks = callbacks)

import pickle as pkl

pkl.dump(model,open("/content/drive/MyDrive/ml_models/autopilot.pkl","wb"))

model.save("/content/drive/MyDrive/ml_models/autopilot.h5")
model.save_weights("/content/drive/MyDrive/ml_models/autopilot.h5")

# Function to calculate mask over image
def weighted_img(img, initial_img, α=1., β=0.5, γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)        # calculates the weighted sum of two arrays...
                                                              # here it is used to adjust contrast and brightness of the image

# Function to process an individual image and it's mask
def process_image_mask(image, mask):
    # Round to closest
    mask = tf.math.round(mask)        # Rounds a value of a Tensor to nearest integer, Rounds the half to even
    
    # Convert to mask image
    zero_image = np.zeros_like(mask)        # Returns an array of zeros with the same shape and type as given array.
    mask = np.dstack((mask, zero_image, zero_image))     # To stack the arrays in sequence depthwise
    mask = np.asarray(mask, np.float32)                  # converts the input to an array of specified dtype
    
    # Convert to image image
    image = np.asarray(image, np.float32)
    
    # Get the final image
    final_image = weighted_img(mask, image)

    return final_image

# Function to save predictions
def save_predictions(dataset):
    # Predict and save image the from input dataset
    
    index = 0
    for batch_image, batch_mask in dataset:
        for image, mask in zip(batch_image, batch_mask):
            print(f"Processing image : {index}")
            pred_mask = model.predict(tf.expand_dims(image, axis = 0))
            save_sample([image, process_image_mask(image, pred_mask[0])], index)
            index += 1

# Function to save the images as a plot
def save_sample(display_list, index):
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.savefig(f"outputs_new_3/{index}.png")
    plt.show()

os.mkdir("outputs_new_3")
save_predictions(dataset['test'])