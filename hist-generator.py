import csv
import numpy as np
from csv_image_generator import csv_image_generator

csv_train_paths = [r'G:\simulator_data\back\cleaned.csv',
                   r'G:\simulator_data\Track1_2_R\cleaned.csv',
                   r'G:\simulator_data\Track1_3\cleaned.csv',
                   r'G:\simulator_data\side\driving_log.csv',
                   ]
                   
csv_train_paths = [#{'folder_path': r"G:\simulator_data\back\\", 'csv_file_name': r'cleaned.csv', 'track': 1}, 
             #{'folder_path': r"G:\simulator_data\Track1_2_R\\", 'csv_file_name': r'cleaned.csv', 'type': 'win', 'track': 1},
             {'folder_path': r"G:\simulator_data\simulator_back\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
             {'folder_path': r"G:\simulator_data\simulator_side\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
             {'folder_path': r"G:\simulator_data\simulator_data\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
             {'folder_path': r"G:\simulator_data\simulator2_back\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
             {'folder_path': r"G:\simulator_data\simulator2_side\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
             {'folder_path': r"G:\simulator_data\simulator2_data\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
             #{'folder_path': r"G:\simulator_data\Track1_3\\", 'csv_file_name': r'cleaned.csv', 'track': 1},
             #{'folder_path': r"G:\simulator_data\side\\", 'csv_file_name': r'driving_log.csv', 'track': 1},
             #{'folder_path': r"G:\simulator_data\data\\", 'csv_file_name': r'driving_log2.csv', 'track': 1},
             ]

csv_valid_paths = [r'G:\simulator_data\Track1_1\driving_log.csv']

count = 20
angles = []

for batch in csv_image_generator(csv_train_paths, BATCH_SIZE=128):

    batch_x = batch[0]
    batch_y = batch[1]
    
    for angle in batch_y:
        angles.append(angle)
    
    count -= 1
    if count == 0:
        break

        
        
import matplotlib.pyplot as plt

binwidth = 0.05

plt.hist(angles, bins=np.arange(min(angles), max(angles) + binwidth, binwidth))
plt.title("Use Center, Left, Right cameras with data augmentation and histogram equalization")
# Use Center, Left, Right cameras with data augmentation and histogram equalization
plt.xlabel("Steering angles")
plt.ylabel("Number of images")
plt.xlim((-1.5, 1.5))
plt.show()