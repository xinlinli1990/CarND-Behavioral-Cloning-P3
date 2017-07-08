import csv
import cv2
import random
import numpy as np
from sklearn.utils import shuffle
from skimage.color import convert_colorspace
from skimage.transform import warp, SimilarityTransform
from skimage.exposure import adjust_gamma

def img_debug(img, info):
    """
    Output debug information
    """
    print(info)
    print('max='+str(np.max(img)))
    print('min='+str(np.min(img)))
    print('dtype='+str(img.dtype))
    print()
    
def get_image_label_from_datum(datum, data_augmentation_params):
    # Unpack datum
    image_path = datum['path']
    measurement = datum['steering']
    horizontal_shift = datum['horizontal_shift_intensity']
    is_flipped = datum['is_flipped']

    # Read image
    image = cv2.imread(image_path)
    
    # Flip it horizontally if needed
    if is_flipped:
        image = cv2.flip(image, flipCode=1)
    
    # Apply image augmentation
    image, measurement = apply_image_augmentation(image, 
                                                  measurement, 
                                                  horizontal_shift,                                                    
                                                  data_augmentation_params)

    return image, measurement
    
def apply_image_augmentation(image, 
                             measurement, 
                             horizontal_shift_intensity,                              
                             data_augmentation_params):
        
    image = random_brightness(image)
    image = random_shadow(image)
    image, measurement = random_shift_transoform(image, 
                                                 measurement, 
                                                 horizontal_shift_intensity, 
                                                 **data_augmentation_params)
    image = np.array(image * 255., dtype=np.uint8)
    
    return image, measurement
    
def random_brightness(img):
    img = convert_colorspace(img, 'RGB', 'HSV')  # 0-255 uint8 -> 0-1 float64
    # img = np.array(img, dtype = np.float64)
    random_bright = 0.5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 1.0] = 1.0
    img = convert_colorspace(img, 'HSV', 'RGB')
    #img = np.array(img * 255., dtype=np.uint8)  # 0-1 float64 -> 0-255 uint8
    return img
    
def random_shadow(img):   
    h, w = img.shape[0], img.shape[1]
    
    left_y = np.random.randint(h/3, h, 2) 
    right_y = np.random.randint(h/3, h, 2)
    left_y = np.sort(left_y)
    right_y = np.sort(right_y)
    
    if left_y[0] == right_y[0] or left_y[1] == right_y[1]:
        return img
    
    img = convert_colorspace(img, 'RGB', 'HSV')  # 0-1 float64
    
    #shadow_mask = np.zeros_like(img[:,:,0])
    X_m = np.mgrid[0:img.shape[0],0:img.shape[1]][1] # x coordinates of mask
    Y_m = np.mgrid[0:img.shape[0],0:img.shape[1]][0] # y coordinates of mask

    # upper line defined by (0, left_y[0]) and (w, right_y[0])
    shadow_mask_1 = ((Y_m - right_y[0]) / (left_y[0] - right_y[0]) - (X_m - w) / (0.0 - w)) <= 0
    # lower line defined by (0, left_y[1]) and (w, right_y[1])
    shadow_mask_2 = ((Y_m - right_y[1]) / (left_y[1] - right_y[1]) - (X_m - w) / (0.0 - w)) >= 0
    
    shadow_mask = (shadow_mask_1 & shadow_mask_2)
    
    img[shadow_mask, 2] *= np.random.uniform() * 0.5 + 0.25
    
    img = convert_colorspace(img, 'HSV', 'RGB')
    
    return img

def random_shift_transoform(img, 
                            measurement, 
                            horizontal_shift_intensity, 
                            horizontal_shift_range, 
                            vertical_shift_range, 
                            steering_angle_per_pixel):
    # Translation
    transform_x = horizontal_shift_range * horizontal_shift_intensity
    transform_y = vertical_shift_range * np.random.uniform() - vertical_shift_range / 2
    measurement += transform_x * steering_angle_per_pixel
    tform = SimilarityTransform(translation=(transform_x, transform_y))
    img = warp(img, tform)
    return img, measurement
    
def histogram_equalization(data, data_augmentation_params, bounds=[-1.5, 1.5], bin_width=0.01, max_datum_per_bin=50):

    output_data = []

    for i in np.arange(bounds[0], bounds[1], bin_width):
        lower_bound = i
        upper_bound = i + bin_width
        count = 0
        
        for datum in data:
            # Compute the actual steering angle after image augmentation
            shift_pixs = datum['horizontal_shift_intensity'] * data_augmentation_params['horizontal_shift_range']
            shift_angle = shift_pixs * data_augmentation_params['steering_angle_per_pixel']
            steering_angle = datum['steering'] + shift_angle
            
            if bounds[0] == 0:
                steering_angle = abs(steering_angle)
               
            if steering_angle >= lower_bound and steering_angle < upper_bound:
                output_data.append(datum)
                count += 1
                
                if count >= max_datum_per_bin:
                    break

    return output_data
    
def csv_image_generator(csv_paths, BATCH_SIZE=128):
    """
    keras data generator
    """

    # Read all csv records into lines[]
    lines = []
    track_count = [0,0,0]
    track_count[1] = 0
    track_count[2] = 0
    
    for csv_path in csv_paths:
        csv_file_name = csv_path['csv_file_name']
        csv_folder_path = csv_path['folder_path']
        csv_file_path = csv_folder_path + csv_file_name
        
        csv_path_separator = '\\'
        if csv_path['type'] == 'mac':
            csv_path_separator = '/'
        elif csv_path['type'] == 'win':
            csv_path_separator = '\\'
            
        csv_track = csv_path['track']
        
        with open(csv_file_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                # Set images path to csv file path
                
                fname_center = line[0].split(csv_path_separator)[-1]
                fname_left = line[1].split(csv_path_separator)[-1]
                fname_right = line[2].split(csv_path_separator)[-1]
                
                line[0] = csv_folder_path + 'IMG\\' + fname_center
                line[1] = csv_folder_path + 'IMG\\' + fname_left
                line[2] = csv_folder_path + 'IMG\\' + fname_right
                
                lines.append(line)
                track_count[csv_track] += 1
    
    # print("Track 1 log : " , track_count[1])
    # print("Track 2 log : " , track_count[2])
    
    data_augmentation_params = {
                                'steering_angle_per_pixel': 0.005,
                                'horizontal_shift_range': 30, 
                                'vertical_shift_range': 30,
                                }
                                
    data_balance_params = {
                           'bounds': [-1.5, 1.5],
                           'bin_width': 0.04,
                           'max_datum_per_bin': 1000,
                           }
                                
    side_camera_shift_pixels = 40
    side_camera_correction = data_augmentation_params['steering_angle_per_pixel'] * side_camera_shift_pixels
    data = []
    
    for line in lines:
        # Get image path for center, left, right cameras
        path_center = line[0]
        path_left = line[1]
        path_right = line[2]

        # Get steering angle for center camera and compute the corresponding angle for left, right cameras
        steering_center = float(line[3])
        steering_left = steering_center + side_camera_correction
        steering_right = steering_center - side_camera_correction
        
        center = {'path': path_center, 
                  'steering': steering_center, 
                  'horizontal_shift_intensity': np.random.uniform() - 0.5,  # -0.5 ~ 0.5
                  'is_flipped': False}
        left = {'path': path_left, 
                'steering': steering_left, 
                'horizontal_shift_intensity': np.random.uniform() - 0.5,
                'is_flipped': False}
        right = {'path': path_right, 
                 'steering': steering_right, 
                 'horizontal_shift_intensity': np.random.uniform() - 0.5,
                 'is_flipped': False}           

        # Add horizontal flipped images and their steering angles
        flip_center = {'path': path_center,      
                       'steering': -1 * steering_center, 
                       'horizontal_shift_intensity': np.random.uniform() - 0.5,
                       'is_flipped': True}
        flip_left = {'path': path_left, 
                     'steering': -1 * steering_left, 
                     'horizontal_shift_intensity': np.random.uniform() - 0.5,
                     'is_flipped':True}
        flip_right = {'path': path_right, 
                      'steering': -1 * steering_right, 
                      'horizontal_shift_intensity': np.random.uniform() - 0.5,
                      'is_flipped':True}     

        data.append(center)
        data.append(left)   
        data.append(right)
        data.append(flip_center)
        data.append(flip_left)
        data.append(flip_right)        

    # Inifite loop for yield
    while 1:        
        # Shuffle data
        data = shuffle(data)

        # Balance data with histogram equalization
        equ_data = histogram_equalization(data, data_augmentation_params, **data_balance_params)
        equ_data = shuffle(equ_data)
        
        # Generate one batch
        num_examples = len(equ_data)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_data = equ_data[offset:end]
            
            # Read images, flip it if necessary, then image augmentation
            batch_images = []
            batch_measurements = []
            
            for datum in batch_data:
                single_image, single_measurement = get_image_label_from_datum(datum, data_augmentation_params)
                batch_images.append(single_image)
                batch_measurements.append(single_measurement)

            batch_x = np.array(batch_images)
            batch_y = np.array(batch_measurements)

            # batch_x, batch_y = shuffle(batch_x, batch_y)
            
            yield [batch_x, batch_y]
