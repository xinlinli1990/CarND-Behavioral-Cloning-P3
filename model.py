import csv
import cv2
import numpy as np
from csv_image_generator import csv_image_generator
from data import csv_train_paths, csv_valid_paths

BATCH_SIZE = 128
                  
# Get train data size                   
lines = []
for csv_path in csv_train_paths:
    csv_file_name = csv_path['csv_file_name']
    csv_folder_path = csv_path['folder_path']
    csv_file_path = csv_folder_path + csv_file_name
        
    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

# Calculate number of batches per EPOCH            
X_train_size = len(lines) * 3 # each line contains three images
steps_per_epoch = int(X_train_size / BATCH_SIZE)

# Get validation data size
lines = []
for csv_path in csv_valid_paths:
    csv_file_name = csv_path['csv_file_name']
    csv_folder_path = csv_path['folder_path']
    csv_file_path = csv_folder_path + csv_file_name

    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

# Calculate number of batches per EPOCH            
X_valid_size = len(lines) * 3
validation_steps = int(X_valid_size / BATCH_SIZE)

# Train the model with Keras            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20), (0,0))))
model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# Use MSE loss and Adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(csv_image_generator(csv_train_paths, BATCH_SIZE=BATCH_SIZE), 
                    steps_per_epoch=steps_per_epoch,
                    validation_data=csv_image_generator(csv_valid_paths, BATCH_SIZE=BATCH_SIZE),
                    validation_steps=validation_steps,
                    epochs=7)

# Save model
model.save('model.h5')