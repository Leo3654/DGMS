import os
import torch

# Define the directory path
dir_path = '/path/to/your/folder'

# Iterate over all the files in the directory
for file_name in os.listdir(dir_path):
    # Check if the file is a .pt file
    if file_name.endswith('.pt'):
        # Load the file
        file_path = os.path.join(dir_path, file_name)
        data = torch.load(file_path)
        
        # Convert the data to a dictionary
        data_dict = dict(data)
        
        # Save the dictionary as _dict.pt file
        dict_file_name = file_name.replace('.pt', '_dict.pt')
        dict_file_path = os.path.join(dir_path, dict_file_name)
        torch.save(data_dict, dict_file_path)