import concurrent.futures
import json
import os

import torch

from ParsePythonGraph import *
from ParseTextGraph import *

glove = GLoVe("./glove.840B.300d.txt")


def work(n):
    id = n
    end = n + 500
    if n == 20500:
        end = 20604
    for data in datas[n:end]:
        codePro = data['code']
        retrieval_idx = data['retrieval_idx']
        for code, code_id in codes.items():
            if code_id == retrieval_idx:
                codePro = code
                break

        try:
            code_graph = PythonGraph(codePro)
        except:
            print(id)
            continue

        id += 1


with open('./cosqa.json', 'r') as f, open('./code_idx_map_new.json', 'r') as f1:
    datas = json.load(f)
    codes = json.load(f1)

work(500)
