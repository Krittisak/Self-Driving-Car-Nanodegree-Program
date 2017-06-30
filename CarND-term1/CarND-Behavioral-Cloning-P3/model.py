import csv
import cv2
import numpy as np
import sklearn

# Prepare data
samples = []
correction = [0.0, 0.2, -0.2]
with open ('./data/driving_log.csv') as csvfile :
	reader = csv.reader (csvfile)
	for line in reader :
		for i in range (3) :
			samples.append ((line[i], False, float (line[3]) + correction[i]))
			samples.append ((line[i], True, (float (line[3]) + correction[i]) * -1.0))

# Split to train and validation
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split (samples, test_size = 0.2)

def generator (samples, batch_size = 32) :
	num_samples = len (samples)
	while 1 :
		sklearn.utils.shuffle (samples)
		for offset in range (0, num_samples, batch_size) :
			batch_samples = samples[offset: offset + batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples :
				name = './data/IMG/' + batch_sample[0].split ('/')[-1]
				image = cv2.imread (name)
				if batch_sample[1] :
					image = cv2.flip (image, 1)
				angle = batch_sample[2]
				images.append (image)
				angles.append (angle)

			X_train = np.array (images)
			y_train = np.array (angles)
			yield sklearn.utils.shuffle (X_train, y_train)

# Data generator
train_generator = generator (train_samples, batch_size = 32)
validation_generator = generator (validation_samples, batch_size = 32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Create model
model = Sequential ()
model.add (Lambda (lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add (Cropping2D (cropping = ((70, 25), (0, 0))))
model.add (Convolution2D (24, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add (Convolution2D (36, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add (Convolution2D (48, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add (Convolution2D (64, 3, 3, activation = 'relu'))
model.add (Convolution2D (64, 3, 3, activation = 'relu'))
model.add (Flatten ())
model.add (Dropout (0.5))
model.add (Dense (100))
model.add (Dropout (0.5))
model.add (Dense (50))
model.add (Dropout (0.5))
model.add (Dense (10))
model.add (Dropout (0.5))
model.add (Dense (1))

# Compile model
model.compile (loss = 'mse', optimizer = 'adam')
model.fit_generator (train_generator, samples_per_epoch = \
	len (train_samples), validation_data = validation_generator, \
	nb_val_samples = len (validation_samples), nb_epoch = 3)

# Save model
model.save ('model.h5')
exit ()