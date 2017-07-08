import csv
import numpy as np

csv_train_paths = [r'G:\simulator_data\back\cleaned.csv',
                   r'G:\simulator_data\Track1_2_R\cleaned.csv',
                   r'G:\simulator_data\Track1_3\cleaned.csv',
                   r'G:\simulator_data\side\driving_log.csv',
                   ]

csv_valid_paths = [r'G:\simulator_data\Track1_1\driving_log.csv']

# Get train size                   
angles = []
for csv_path in csv_train_paths:
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            angles.append(float(line[3]))

import matplotlib.pyplot as plt

binwidth = 0.01

plt.hist(angles, bins=np.arange(min(angles), max(angles) + binwidth, binwidth))
plt.show()