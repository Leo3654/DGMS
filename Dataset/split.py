import json
import os

split_ids = {}
split_ids["train"] = []
split_ids["valid"] = []
split_ids["test"] = []
with open('./csnPro/split.json', 'w') as f:
    for i in range(21000, 401598):
        if os.path.exists("./csnPro/code_processed/code_" + str(i) + ".pt"):
            split_ids["train"].append(i)
    for i in range(0, 20000):
        if os.path.exists("./csnPro/code_processed/code_" + str(i) + ".pt"):
            split_ids["valid"].append(i)
    for i in range(20000, 21000):
        if os.path.exists("./csnPro/code_processed/code_" + str(i) + ".pt"):
            split_ids["test"].append(i)
    # split_ids["valid"] = list(range(0, 497))
    # split_ids["test"] = list(range(500, 998))
    json.dump(split_ids, f)
    print(len(split_ids["train"]))
    print(len(split_ids["valid"]))
    print(len(split_ids["test"]))
