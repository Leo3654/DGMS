from LoadGLoVe import GLoVe

from natsort import natsorted

import os
import json

directory = '../../python/final/jsonl/'
total_count = 0

def process_one_line(line):
    json = json.loads(line)
    docstring = json['docstring']
    code = json['code']

    

for filename in natsorted(os.listdir(directory)):
    if filename.endswith('.jsonl'):
        with open(os.path.join(directory, filename), 'r') as f:
            count = 0
            for line in f:
                process_one_line(line)
            total_count += count
        print(f'{filename}: {count}')

print(f'Total count: {total_count}')