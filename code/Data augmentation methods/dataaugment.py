import keras
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import scipy.stats as stat
from scipy.io import loadmat
from sklearn import preprocessing
import cv2
import scipy.io as sio
from utils import load_file, save_file
import glob
from os import listdir
from os.path import isfile, join
from preprocessing import extract_flat_images
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
df=load_file('data/solar_panel/train_fail_images.pkl')
print(df)
images = np.reshape(df, (df.shape[0], int(np.sqrt(df.shape[1])), int(np.sqrt(df.shape[1]))))
fig, ax = plt.subplots(nrows=1, ncols=32, figsize=(50,50))


for i in range(16):
	# plot image
	ax[i].imshow(images[i])
	ax[i].axis('off')

images = np.expand_dims(images, axis=-1)

#############
#image = tf.keras.preprocessing.image.load_img(train_fails)
#input_arr = keras.preprocessing.image.img_to_array(image)
#input_arr = np.array([input_arr])  # Convert single image to a batch.
#predictions = model.predict(input_arr)
#augment=ImageDataGenerator()
augment = ImageDataGenerator(
     	rotation_range=[45,90,135,180,225,270],
		width_shift_range=0.5,
		height_shift_range=0.5,
		#shear_range=0.10,
		horizontal_flip=True,
        vertical_flip=True)

data=augment.flow(images, batch_size=16)

#fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(100,100))

for i in range(29):
	x = next(data)[0].astype('uint8')
	x = np.squeeze(x, axis=-1)
	# plot image
	ax[i+3].imshow(x)
	ax[i+3].axis('off')
plt.show()

#try:
#    inputReader = csv.reader(open(argv[1], encoding='ISO-8859-1'), delimiter=',',quotechar='"')
#except IOError:
#    pass
#ImageDataGenerator.flow_from_dataframe(
    #train_fails,
    #directory=None,
    #x_col="filename",
    #y_col="class",
    #weight_col=None,
    #target_size=(100, 100),
    #color_mode="grayscale",
    #class_mode="categorical",
    #batch_size=64,
    #shuffle=True,
    #seed=None,
    #save_to_dir=('data/augmentedimages'),
    #save_format="png",
    #interpolation="nearest",
#)
#aug_it=augment.flow(img, batch_size=64)
# generate batch of images
#for i in range(3):

	# convert to unsigned integers
	#image = next(aug_it)[0].astype('uint8')


#augment = tf.keras.preprocessing.image.array_to_img(train_fails)

#aug=dataaugment(augment)
