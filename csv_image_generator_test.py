from csv_image_generator import csv_image_generator
import cv2
import matplotlib.pyplot as plt

csv_paths = [#{'folder_path': r"G:\simulator_data\back\\", 'csv_file_name': r'cleaned.csv', 'track': 1}, 
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
             

count = 5
for batch in csv_image_generator(csv_paths):

    batch_x = batch[0]
    batch_y = batch[1]
    
    img = cv2.cvtColor(batch_x[count], cv2.COLOR_BGR2RGB)
    plt.imsave(str(count) + ".png", img)
    # plt.imshow(img)
    # plt.title("Steering angle = " + str(batch_y[count]))
    # plt.show()
    
    count -= 1
    if count == 0:
        break
        
        
# import numpy as np

# def balanced_data(img_paths, measurements, bounds=[-1.0, 1.0], bins=100, max=200):

    # step = (bounds[1] - bounds[0]) / (bins - 1)
    
    # output_img_paths = []
    # output_measurements = []

    # for i in np.arange(bounds[0], bounds[1], step):
        # lower_bound = i
        # upper_bound = i + step
        # count = 0
        
        # for img_path, measurement in zip(img_paths, measurements):
            # if measurement >= lower_bound and measurement < upper_bound:
                # output_img_paths.append(img_path)
                # output_measurements.append(measurement)
                # count += 1
                
                # if count >= max:
                    # break

    # return output_img_paths, output_measurements
    

# img_paths = [1, 2, 3, 4, 5, 6, 7]
# measurements = [0.0, 0.0, 0.0,0.0, 0.11]
 
# img_paths, measurements = balanced_data(img_paths, measurements, max=2)

# print(measurements)