from LoadGLoVe import GLoVe

from ParseTextGraph import *
from ParsePythonGraph import *

from datetime import datetime

from natsort import natsorted

import torch

import os
import json

import multiprocessing as mp

import pycld2 as cld2

from multiprocessing.dummy import Pool as ThreadPool


directory = '../../python/final/jsonl'
max_nodes = 300
# Get parent directory of the directory
parent_dir = os.path.dirname(directory)

# Get the current date and time as a string 
date_time = datetime.now().strftime("%D/%H-%M-%S")

processed_dir = parent_dir+"/processed/" + date_time

text_dir = processed_dir + "/text_processed"
if not os.path.exists(text_dir):
    os.makedirs(text_dir) 

code_dir = processed_dir + "/code_processed"
if not os.path.exists(code_dir):
    os.makedirs(code_dir)

original_code_dir = processed_dir + "/original_code"
if not os.path.exists(original_code_dir):
    os.makedirs(original_code_dir)

code_graph_ids = []
text_graph_ids = []
total_count = 0

glove = GLoVe("../../glove.840B.300d.txt")

def process_one_line(data, id, last_docstring = None):
    docstring = data['docstring']
    #print("Docstring:", docstring)
    code = data['code']

    if last_docstring == docstring:
        print("========= Skipping, duplicate ===========")
        data["error"] = "duplicate"
        with open(original_code_dir + f'/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False
    
    try:
        _, _, details = cld2.detect(docstring)
    except:
        print("Skipping, docstring error")
        data["error"] = "docstring error"
        with open(original_code_dir + f'/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False

    language = details[0][1]
    certainty = details[0][2]

    #If not English or certainty is less than 0.5, return false
    if language != "en" or certainty <= 60:
        print("Skipping, non-English")
        data["error"] = "non-English"
        with open(original_code_dir + f'/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False
    

    try: 
        code_graph = PythonGraph(code)
    except:
        print("Skipping, code error")
        data["error"] = "code error"
        return False

    code_graph.add_word_ordering_edges(next_word = False)
    code_graph.add_last_lexical_use_edges()
    code_pyg = code_graph.convert_to_pyg(glove)

    code_graph_nodes = code_graph.num_nodes()

    print("Code nodes:", code_graph_nodes)

    if code_graph_nodes > max_nodes:
        print("Skipping, cg")
        data["error"] = "cg too large"
        with open(original_code_dir + f'/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False

    try:
        text_graph = DocstringGraph(docstring)
    except:
        print("Skipping, text error")
        data["error"] = "text error"
        with open(original_code_dir + f'/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False
    text_graph.add_word_ordering_edges()
    text_pyg = text_graph.convert_to_pyg(glove)
    text_graph_nodes = text_graph.num_nodes()
    print("Text nodes:", text_graph_nodes)

    if text_graph_nodes > max_nodes:
        print("Skipping, tg")
        data["error"] = "tg too large"
        with open(original_code_dir + f'/code_{id}.json', 'w') as f:
            json.dump(data, f)
        return False

    torch.save(text_pyg, text_dir+f'/text_{id}.pt')
    torch.save(code_pyg, code_dir+f'/code_{id}.pt')
    
    data["num_text_nodes"] = text_graph_nodes
    data["num_code_nodes"] = code_graph_nodes

    # Save json file
    with open(original_code_dir + f'/code_{id}.json', 'w') as f:
        json.dump(data, f)

    return True

def process_one_file(filename_and_id):
    ids = []

    filename, id = filename_and_id
    with open(os.path.join(dir, filename), 'r') as f:
        docstring = None
        for line in f:
            data = json.loads(line)
            print("Processing", id, os.path.join(dir, filename))
            try:
                if process_one_line(data, id, docstring):
                    ids.append(id)
                docstring = pre_process_docstring(data['docstring'])
            except:
                print("Skipping, error")
                data["error"] = "error"
                with open(original_code_dir + f'/code_{id}.json', 'w') as f:
                    json.dump(data, f)
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
with open(processed_dir + '/split.json', 'w') as f:
    json.dump(split, f)

print(f'Total count: {total_count}')

print("Finish time:", datetime.now().strftime("%D %H:%M:%S"))
