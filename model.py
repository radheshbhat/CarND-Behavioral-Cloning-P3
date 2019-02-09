import tensorflow as tf
#tf.python.control_flow_ops = tf

from keras.models import Sequential, model_from_json, load_model
from keras.optimizers import *
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D

from scipy.misc import imread, imsave
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 10
PATH = "data/"
CSV_FILE = "driving_log.csv"

DATA = pd.read_csv(PATH + CSV_FILE, usecols = [0, 1, 2, 3])

TRAINING_DATA, VALIDATION_DATA = train_test_split(DATA, test_size = 0.15)
TOTAL_TRAIN_DATA = len(TRAINING_DATA)
TOTAL_VALID_DATA = len(VALIDATION_DATA)

#Flipping the image and changing the sign of steering angle 
def flipped(image, measurement):
  return np.fliplr(image), -measurement

def get_image(i, data):

  positions, corrections = ['left', 'center', 'right'], [.25, 0, -.25]
  ID, r = data.index[i], random.choice([0, 1, 2])

  measurement = data['steering'][ID] + corrections[r]

  path = PATH + data[positions[r]][ID][1:]
  if r == 1: path = PATH + data[positions[r]][ID]
  image = imread(path)

  if random.random() > 0.5:
    image, measurement = flipped(image, measurement)

  return image, measurement


def generate_data(data, batch_size):

  while True:

    SIZE = len(data)
    data.sample(frac = 1)

    for start in range(0, SIZE, batch_size):
      images, measurements = [], []

      for this_id in range(start, start + batch_size):
        if this_id < SIZE:
          image, measurement = get_image(this_id, data)
          measurements.append(measurement)
          images.append(image)

      yield np.array(images), np.array(measurements)

# Reference : https://github.com/commaai/research/blob/master/train_steering_model.py

model = Sequential()

#Normalize the data.
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))

# Crop image to only see section with road
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Convolution2D(16, 8, 8, subsample = (4, 4), border_mode = "same"))

#Exponential Linear Units
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample = (2, 2), border_mode = "same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.summary()
# Use mean squared error for regression, and an Adams optimizer.
model.compile(optimizer = "adam", loss = "mse")



### Experiment with Nvidia model
#model = Sequential()
#model.add( Cropping2D( cropping=( (50,20), (0,0) ), input_shape=(160,320,3)))
#model.add( Lambda( lambda x: x/255. - 0.5 ) )
#model.add( Convolution2D( 24, 5, 5, subsample=(2,2), activation = 'relu' ) )
#model.add( Convolution2D( 36, 5, 5, subsample=(2,2), activation = 'relu' ) )
#model.add( Convolution2D( 48, 5, 5, subsample=(2,2), activation = 'relu' ) )
#model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
#model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
#model.add( Flatten() )
#model.add( Dense( 100 ) )
#model.add(Dropout(0.5)) 
#model.add( Dense( 50 ) )
#model.add( Dense( 10 ) )
#model.add( Dense( 1 ) )
#model.compile( loss='mse', optimizer='adam' )




print('Training the model...')

training_generator = generate_data(TRAINING_DATA, batch_size = BATCH_SIZE)
validation_generator = generate_data(VALIDATION_DATA, batch_size = BATCH_SIZE)


history_object = model.fit_generator(training_generator,
                 samples_per_epoch = TOTAL_TRAIN_DATA,
                 validation_data = validation_generator,
                 nb_val_samples = TOTAL_VALID_DATA,
                 nb_epoch = NUMBER_OF_EPOCHS,
                 #callbacks = [early_stopping],
                 verbose = 1)

print('Saving the model...')

model.save("model.h5")

with open("model.json", "w") as json_file:
  json_file.write(model.to_json())

print("Model is saved.")
  
# Plot the training and validation loss for each epoch
print('Generating loss chart...')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model.png')  

