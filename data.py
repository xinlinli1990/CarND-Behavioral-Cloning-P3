
# folder_path: csv file folder path, image location is folder_path\IMG\*.jpg
# csv_file_name: csv file name
# type: 'mac': data collected on MacOS or Linux, path splited by '/', 
#       'win': data collected on Windows, path splited by '\'
# track: data collected from track1 or track2

# Define training data set
csv_train_paths = [#{'folder_path': r"G:\simulator_data\back\\", 'csv_file_name': r'cleaned.csv', 'type': 'win', 'track': 1}, 
                   #{'folder_path': r"G:\simulator_data\Track1_2_R\\", 'csv_file_name': r'cleaned.csv', 'type': 'win', 'track': 1},
                   #{'folder_path': r"G:\simulator_data\Track1_3\\", 'csv_file_name': r'cleaned.csv', 'type': 'win', 'track': 1},
                   #{'folder_path': r"G:\simulator_data\side\\", 'csv_file_name': r'driving_log.csv', 'type': 'win', 'track': 1},
                   #{'folder_path': r"G:\simulator_data\data\\", 'csv_file_name': r'driving_log2.csv', 'type': 'win', 'track': 1},
                   {'folder_path': r"G:\simulator_data\simulator_back\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
                   {'folder_path': r"G:\simulator_data\simulator_side\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
                   {'folder_path': r"G:\simulator_data\simulator_data\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 1},
                   {'folder_path': r"G:\simulator_data\simulator2_back\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
                   {'folder_path': r"G:\simulator_data\simulator2_side\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
                   {'folder_path': r"G:\simulator_data\simulator2_data\\", 'csv_file_name': r'driving_log.csv', 'type': 'mac', 'track': 2},
                   ]
                   
# Define validation data set
csv_valid_paths = [{'folder_path': r"G:\simulator_data\Track1_1\\", 'csv_file_name': r'driving_log.csv', 'type': 'win', 'track': 1}, ]
