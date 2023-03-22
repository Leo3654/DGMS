import os
import torch
import sys

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
        
        # Convert the data to a dictionary
        data_dict = data.to_dict()
        
        # Save the dictionary as _dict.pt file
        dict_file_name = file_name.replace('.pt', '_dict.pt')
        dict_file_path = os.path.join(dict_dir_path, dict_file_name)
        torch.save(data_dict, dict_file_path)
