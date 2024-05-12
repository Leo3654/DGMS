import concurrent.futures
import json
import os

import torch

from ParsePythonGraph import *
from ParseTextGraph import *

processed_dir = './VarCosqa'
text_dir = processed_dir + "/text_processed"
if not os.path.exists(text_dir):
    os.makedirs(text_dir)

code_dir = processed_dir + "/code_processed"
if not os.path.exists(code_dir):
    os.makedirs(code_dir)

glove = GLoVe("./glove.840B.300d.txt")


def work(n):
    id = n
    end = n + 500
    if n == 20500:
        end = 20604
    for data in datas[n:end]:
        docstring = data['doc']
        codePro = data['code']
        retrieval_idx = data['retrieval_idx']
        for code, code_id in codes.items():
            if code_id == retrieval_idx:
                codePro = code
                break

        try:
            code_graph = PythonGraph(codePro)
        except:
            continue

        code_graph.add_word_ordering_edges(next_word=False)
        code_graph.add_last_lexical_use_edges()
        code_pyg = code_graph.convert_to_pyg(glove)

        text_graph = DocstringGraph(docstring)
        text_graph.add_word_ordering_edges()
        text_pyg = text_graph.convert_to_pyg(glove)

        torch.save(text_pyg, text_dir + f'/text_{id}.pt')
        torch.save(code_pyg, code_dir + f'/code_{id}.pt')

        id += 1

    if n == 0 or n == 500:
        os.makedirs(processed_dir + '/' + str(id - 1))


with open('./cosqa.json', 'r') as f, open('./code_idx_map_new.json', 'r') as f1:
    datas = json.load(f)
    codes = json.load(f1)

# Number of items: 20604
# 创建一个任务列表
tasks = list(range(0, 21000, 500))

# 使用ThreadPoolExecutor并行处理任务
with concurrent.futures.ThreadPoolExecutor(max_workers=42) as executor:
    future_to_result = {executor.submit(work, task): task for task in tasks}
    for future in concurrent.futures.as_completed(future_to_result):
        task = future_to_result[future]
        print(f"Task {task} finished")
