import os
import torch

path = '/var/tmp/python-csn-new/text_processed'
files = os.listdir(path)
total_files = len(files)

for i, file in enumerate(files):
    data = torch.load(os.path.join(path, file))
    data.edge_attr = data.edge_attr.float()
    data.x = data.x.float()
    torch.save(data, os.path.join(path, file))
    if i % 100 == 0:
        print(f'{i/total_files*100:.2f}% done')