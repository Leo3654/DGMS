from LoadGLoVe import GLoVe
from ParseTextGraph import *
from ParsePythonGraph import *
from datetime import datetime
from natsort import natsorted
import torch
import os
import multiprocessing as mp
import pycld2 as cld2
from multiprocessing.dummy import Pool as ThreadPool
import json
import re

processed_dir = './p'
text_dir = processed_dir + "/text_processed"
if not os.path.exists(text_dir):
    os.makedirs(text_dir)

code_dir = processed_dir + "/code_processed"
if not os.path.exists(code_dir):
    os.makedirs(code_dir)

glove = GLoVe("./glove.840B.300d.txt")

with open('cosqa-retrieval-train-19604.json', 'r') as f:
    datas = json.load(f)

id = 1000

for data in datas[0:1000]:
    docstring = data['doc']
    codeOld = data['code']
    code = re.sub(r'"""[\d\D]*"""[\s]+', '', codeOld)
    code_graph = PythonGraph(code)

    code_graph.add_word_ordering_edges(next_word=False)
    code_graph.add_last_lexical_use_edges()
    code_pyg = code_graph.convert_to_pyg(glove)
    code_graph_nodes = code_graph.num_nodes()

    text_graph = DocstringGraph(docstring)
    text_graph.add_word_ordering_edges()
    text_pyg = text_graph.convert_to_pyg(glove)
    text_graph_nodes = text_graph.num_nodes()

    torch.save(text_pyg, text_dir + f'/text_{id}.pt')
    torch.save(code_pyg, code_dir + f'/code_{id}.pt')

    id += 1
