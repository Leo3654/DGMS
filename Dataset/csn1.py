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
import concurrent.futures

processed_dir = './csnPro'
text_dir = processed_dir + "/text_processed"
if not os.path.exists(text_dir):
    os.makedirs(text_dir)

code_dir = processed_dir + "/code_processed"
if not os.path.exists(code_dir):
    os.makedirs(code_dir)

glove = GLoVe("./glove.840B.300d.txt")

with open('csn_train.json', 'r') as f:
    datas = json.load(f)

# Number of items: 401598
# 创建一个任务列表
tasks = [23, 28, 30, 31, 32, 36, 38, 39, 41, 42, 43, 47, 51, 53, 55, 57, 59, 61, 63, 64, 66, 68, 70, 72, 74, 75, 76, 78,
         80, 81, 82, 86, 90, 91, 93, 95, 100, 101, 102, 103, 104, 110, 111, 113, 114, 115, 116, 117, 119, 122, 123, 124,
         125, 127, 128, 129, 130, 131, 132, 133, 137, 139, 140, 141, 142, 143, 145, 150, 151, 152, 153, 156, 157, 159,
         160, 162, 163, 164, 166, 168, 170, 171, 172, 174, 176, 177, 178, 179, 180, 181, 185, 186, 191, 193, 194, 196,
         198, 199, 203, 204, 205, 206, 207, 209, 211, 213, 215, 216, 217, 221, 222, 223, 225, 227, 228, 229, 230, 231,
         232, 236, 237, 238, 241, 244, 245, 246, 248, 249, 250, 251, 252, 254, 258, 259, 262, 263, 264, 266, 268, 271,
         272, 273, 276, 279, 280, 283, 287, 288, 291, 295, 297, 300, 301, 302, 304, 305, 307, 313, 314, 317, 319, 321,
         322, 325, 327, 328, 329, 331, 332, 333, 335, 336, 338, 339, 341, 342, 344, 345, 347, 348, 350, 351, 353, 357,
         358, 359, 360, 362, 365, 367, 368, 369, 371, 372, 376, 380, 381, 382, 383, 384, 385, 388, 390, 391, 393, 395,
         396, 398, 400]


def work(n):
    id = n * 1000

    for data in datas[n * 1000:n * 1000 + 1000]:
        docstring = data['doc']
        code = data['code']
        code_graph = PythonGraph(code)

        code_graph.add_word_ordering_edges(next_word=False)
        code_graph.add_last_lexical_use_edges()
        code_pyg = code_graph.convert_to_pyg(glove)

        text_graph = DocstringGraph(docstring)
        text_graph.add_word_ordering_edges()
        text_pyg = text_graph.convert_to_pyg(glove)

        torch.save(text_pyg, text_dir + f'/text_{id}.pt')
        torch.save(code_pyg, code_dir + f'/code_{id}.pt')

        id += 1


# 使用ThreadPoolExecutor并行处理任务
with concurrent.futures.ThreadPoolExecutor(max_workers=209) as executor:
    future_to_result = {executor.submit(work, task): task for task in tasks}
    for future in concurrent.futures.as_completed(future_to_result):
        task = future_to_result[future]
        print(f"Task {task} finished")
