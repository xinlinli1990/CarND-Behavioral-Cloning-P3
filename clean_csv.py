import csv
import cv2
import numpy as np

csv_paths = [r'G:\simulator_data\back\driving_log.csv',
             r'G:\simulator_data\Track1_1\driving_log.csv',
             r'G:\simulator_data\Track1_2_R\driving_log.csv',
             r'G:\simulator_data\Track1_3\driving_log.csv',
             ]

for csv_path in csv_paths:
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            source_path = line[0]
            filename = source_path.split('/')[-1]
            current_path = source_path#'current_path' + filename
            image = cv2.imread(current_path)
            
            if image is None:
                continue
                
            lines.append(line)
    csv_path_new = '\\'.join(csv_path.split('\\')[0:-1]) + r"\cleaned.csv"
    
    with open(csv_path_new, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for line in lines:
            writer.writerow([str(line[0]),
                             str(line[1]), 
                             str(line[2]), 
                             str(line[3]), 
                             str(line[4]), 
                             str(line[5]), 
                             str(line[6])])
            
# Remove invalid data
valid_lines = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = source_path#'current_path' + filename
    image = cv2.imread(current_path)
    
    # If image read FAILED, skip it

    valid_lines.append(line)
lines = valid_lines