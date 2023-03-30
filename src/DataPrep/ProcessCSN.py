from LoadGLoVe import GLoVe

from ParseTextGraph import *
from ParsePythonGraph import *

from datetime import datetime

from natsort import natsorted

import torch

import os
import json

import multiprocessing as mp

from langdetect import detect
from langdetect import detect_langs

from multiprocessing.dummy import Pool as ThreadPool


directory = '../../python/final/jsonl'
max_nodes = 300
# Get parent directory of the directory
parent_dir = os.path.dirname(directory)

text_dir = parent_dir+"/processed/text_processed"
if not os.path.exists(text_dir):
    os.makedirs(text_dir) 

code_dir = parent_dir+"/processed/code_processed"
if not os.path.exists(code_dir):
    os.makedirs(code_dir)

original_code_dir = parent_dir+"/processed/original_code"
if not os.path.exists(original_code_dir):
    os.makedirs(parent_dir+"/processed/original_code/")

code_graph_ids = []
text_graph_ids = []
total_count = 0

glove = GLoVe("../../glove.840B.300d.txt")

def process_one_line(line, id):
    data = json.loads(line)
    docstring = data['docstring']
    #print("Docstring:", docstring)
    code = data['code']

    
    #If not English or certainty is less than 0.5, return false
    if detect(docstring) != 'en' or detect_langs(docstring)[0].prob < 0.6:
        print("Skipping, non-English")
        data["error"] = "non-English"
        with open(parent_dir + f'/processed/original_code/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False
    

    code_graph = PythonGraph(code)
    code_graph.add_word_ordering_edges(next_word = False)
    code_graph.add_last_lexical_use_edges()
    code_pyg = code_graph.convert_to_pyg(glove)

    code_graph_nodes = code_graph.num_nodes()

    print("Code nodes:", code_graph_nodes)

    if code_graph_nodes > max_nodes:
        print("Skipping, cg")
        data["error"] = "cg too large"
        with open(parent_dir + f'/processed/original_code/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False

    text_graph = DocstringGraph(docstring)
    text_graph.add_word_ordering_edges()
    text_pyg = text_graph.convert_to_pyg(glove)
    text_graph_nodes = text_graph.num_nodes()
    print("Text nodes:", text_graph_nodes)

    if text_graph_nodes > max_nodes:
        print("Skipping, tg")
        data["error"] = "tg too large"
        with open(parent_dir + f'/processed/original_code/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False

    torch.save(text_pyg, parent_dir + f'/processed/text_processed/text_{id}.pt')
    torch.save(code_pyg, parent_dir + f'/processed/code_processed/code_{id}.pt')
    
    data["num_text_nodes"] = text_graph_nodes
    data["num_code_nodes"] = code_graph_nodes

    # Save json file
    with open(parent_dir + f'/processed/original_code/code_{id}.json', 'w') as f:
        json.dump(data, f)

    return True

def process_one_file(filename_and_id):
    ids = []

    filename, id = filename_and_id
    with open(os.path.join(dir, filename), 'r') as f:
        for line in f:
            print("Processing", id, os.path.join(dir, filename))
            if process_one_line(line, id):
                ids.append(id)
            id += 1

    return ids

#Print current time
print("Start time:", datetime.now().strftime("%D %H:%M:%S"))

#open pt file
#text_pyg = torch.load(parent_dir + f'/processed/text_processed/text_{id}.pt')

split = {}

starting_id = 0
for sample_type in ["train", "valid", "test"]:
    print(sample_type)
    split[sample_type] = []

    dir = directory + "/" + sample_type

    files = natsorted(os.listdir(dir))

    for i in range(len(files)):
        #Get number of lines in file
        files[i] = (files[i], starting_id)
        with open(os.path.join(dir, files[i][0]), 'r') as f:
            for n, l in enumerate(f):
                pass
        print("File:", files[i])
        starting_id += n + 1

    with ThreadPool(mp.cpu_count()) as pool:
        ids = pool.map(process_one_file, files)
        all_ids = sum(ids, [])
        starting_id = max(all_ids) + 1
        split[sample_type] = all_ids
        total_count += len(all_ids)

    #Print current date and time
    print("Time after", sample_type, datetime.now().strftime("%D %H:%M:%S"))

# Write split to json file
with open(parent_dir + '/processed/split.json', 'w') as f:
    json.dump(split, f)

print(f'Total count: {total_count}')

print("Finish time:", datetime.now().strftime("%D %H:%M:%S"))
