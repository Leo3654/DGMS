import os
import torch
import sys

# This file is used to convert the pt file with old format 
# to dictionary file via pyg version 1.7.2
# Example: ConvertPtToDict ../../Datasets/python/code_processed
# Then all pt files under the directory will be converted and saved
# under the directory ../../Datasets/python/code_processed/dict_files

# Define the directory path
dir_path = sys.argv[1]

# Define the directory path for the dictionary files
dict_dir_path = os.path.join(dir_path, 'dict_files')

# Create the directory if it doesn't exist
if not os.path.exists(dict_dir_path):
    os.makedirs(dict_dir_path)

# Iterate over all the files in the directory
for file_name in os.listdir(dir_path):
    # Check if the file is a .pt file
    if file_name.endswith('.pt'):
        # Load the file
        file_path = os.path.join(dir_path, file_name)

        print("Processing", file_name)

        data = torch.load(file_path)
        
        print(data)
        # Convert the data to a dictionary
        data_dict = data.to_dict()
        
        # Save the dictionary as _dict.pt file
        dict_file_name = file_name.replace('.pt', '_dict.pt')
        dict_file_path = os.path.join(dict_dir_path, dict_file_name)
        torch.save(data_dict, dict_file_path)
