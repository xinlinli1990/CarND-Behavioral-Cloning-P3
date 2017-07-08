import csv
import cv2
import numpy as np
from csv_image_generator import csv_image_generator

BATCH_SIZE = 128
                  
csv_train_paths = [#{'folder_path': r"G:\simulator_data\back\\", 'csv_file_name': r'cleaned.csv', 'type': 'win', 'track': 1}, 
                   #{'folder_path': r"G:\simulator_data\Track1_2_R\\", 'csv_file_name': r'cleaned.csv', 'type': 'win', 'track': 1},
                   #{'folder_path': r"G:\simulator_data\Track1_3\\", 'csv_file_name': r'cleaned.csv', 'type': 'win', 'track': 1},
                   #{'folder_path': r"G:\simulator_data\side\\", 'csv_file_name': r'driving_log.csv', 'type': 'win', 'track': 1},
                   #{'folder_path': r"G:\simulator_data\data\\", 'csv_file_name': r'driving_log2.csv', 'type': 'win', 'track': 1},
                   {'folder_path': r"F:\simulator_data\simulator_back\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
                   {'folder_path': r"F:\simulator_data\simulator_side\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
                   {'folder_path': r"F:\simulator_data\simulator_data\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
                   {'folder_path': r"F:\simulator_data\simulator2_back\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
                   {'folder_path': r"F:\simulator_data\simulator2_side\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
                   {'folder_path': r"F:\simulator_data\simulator2_data\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
                   ]

csv_valid_paths = [{'folder_path': r"G:\simulator_data\Track1_1\\", 'csv_file_name': r'driving_log.csv', 'type': 'win', 'track': 1}, ]

# Get train size                   
lines = []
for csv_path in csv_train_paths:
    csv_file_name = csv_path['csv_file_name']
    csv_folder_path = csv_path['folder_path']
    csv_file_path = csv_folder_path + csv_file_name
        
    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
X_train_size = len(lines) * 3
steps_per_epoch = int(X_train_size / BATCH_SIZE)

# Get valid size
lines = []
for csv_path in csv_valid_paths:
    csv_file_name = csv_path['csv_file_name']
    csv_folder_path = csv_path['folder_path']
    csv_file_path = csv_folder_path + csv_file_name

    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
X_valid_size = len(lines) * 3
validation_steps = int(X_valid_size / BATCH_SIZE)
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

# keras.layers.convolutional.Conv2D(filters, 
                                  # kernel_size, 
                                  # strides=(1, 1), 
                                  # padding='valid', 
                                  # data_format=None, 
                                  # dilation_rate=(1, 1), 
                                  # activation=None, 
                                  # use_bias=True, 
                                  # kernel_initializer='glorot_uniform', 
                                  # bias_initializer='zeros', 
                                  # kernel_regularizer=None, 
                                  # bias_regularizer=None, 
                                  # activity_regularizer=None, 
                                  # kernel_constraint=None, 
                                  # bias_constraint=None)
                                  
# keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), 
                                  # strides=None, 
                                  # padding='valid', 
                                  # data_format=None)

#model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

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

# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((60,20), (0,0))))
# model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(1))

# LeNet
#model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((60,20), (0,0))))
# model.add(Conv2D(filters=6, kernel_size=(5,5), activation="relu"))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=6, kernel_size=(5,5), activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(csv_image_generator(csv_train_paths, BATCH_SIZE=BATCH_SIZE), 
                    steps_per_epoch=steps_per_epoch,
                    validation_data=csv_image_generator(csv_valid_paths, BATCH_SIZE=BATCH_SIZE),
                    validation_steps=validation_steps,
                    epochs=60)

model.save('model_60ep.h5')