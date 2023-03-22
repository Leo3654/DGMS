import os
import torch
from torch_geometric.data import Data

def load_and_convert_dicts(folder_path):
    parent_folder_path = os.path.dirname(folder_path)
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_dict.pt"):
            print("Processing", file_name)
            file_path = os.path.join(folder_path, file_name)
            data_dict = torch.load(file_path)
            data = Data.from_dict(data_dict)
            new_file_name = file_name.replace("_dict", "")
            new_file_path = os.path.join(parent_folder_path, new_file_name)
            torch.save(data, new_file_path)

if __name__ == "__main__":
    folder = argv[1]

    load_and_convert_dicts(folder)