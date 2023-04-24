# Deep Graph Matching and Searching for Semantic Code Retrieval

## 1. Description

In this paper, we propose an end-to-end [Deep Graph Matching and Searching (**DGMS**) model](https://dl.acm.org/doi/abs/10.1145/3447571) for the task of semantic code retrieval. Specifically, we first represent both natural
language query texts and programming language codes with the unified graph structured data, and then use the proposed graph matching and searching model to retrieve the best
matching code snippet.
![system](./Model.png)

### 1.1 citation:

Xiang Ling, Lingfei Wu, Saizhuo Wang, Gaoning Pan, Tengfei Ma, Fangli Xu, Alex X. Liu, Chunming Wu, and Shouling Ji, **Deep Graph Matching and Searching for Semantic Code Retrieval**, ACM Transactions on Knowledge Discovery from Data (**TKDD**), 2021, 15(5): 1-21.

 ``` 
  @article{ling2020deep,
    title={Deep Graph Matching and Searching for Semantic Code Retrieval},
    author={Ling, Xiang and Wu, Lingfei and Wang, Saizhuo and Pan, Gaoning and Ma, Tengfei and Xu, Fangli and Liu, Alex X and Wu, Chunming and Ji, Shouling},
    journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
    volume={15},
    number={5},
    publisher={ACM},
    url={https://dl.acm.org/doi/10.1145/3447571},
    year={2021}
  }
 ```

### 1.2 glance:

```
├─── src
│    ├─── model
│    │    ├─── GraphMatchModel.py
│    ├─── config.py
│    ├─── ProcessedDataset.py
│    ├─── train.py
│    ├─── utils.py
├─── Dataset
│    ├─── java
│    │    ├─── code_graph_ids.pt
│    │    ├─── code_processed
│    │    │    ├─── code_100000.pt
│    │    │    ├─── ...
│    │    ├─── text_graph_ids.pt
│    │    ├─── text_processed
│    │    │    ├─── text_100000.pt
│    │    │    ├─── ...
│    │    ├─── split.json
│    ├─── python
│    │    ├─── code_graph_ids.pt
│    │    ├─── code_processed
│    │    │    ├─── code_100000.pt
│    │    │    ├─── ...
│    │    ├─── text_graph_ids.pt
│    │    ├─── text_processed
│    │    │    ├─── text_100000.pt
│    │    │    ├─── ...
│    │    ├─── split.json
```

We build our model on the graph representation of the source code or description text and save each graph representation with ``torch_geometric.data.Data`` (PyTorch_Geometric). To
be specific, each ``code/text_X.pt`` in ``text/code_processed`` folder is the saved graph representation of one text/code graph whose index is `X`. The total size of two datasets
is over 100G, please download the zipped files from Baidu NetDisc and unzip them into the corresponding folder in this repo.

> NetDisc Link: https://pan.baidu.com/s/1CbzQWireoH5hMopK3CRZOw
> Extraction code: 9xz5 (does not work with newer versions of PyTorch)

> Schankula & Li Preprocessed Dataset: [Download Link](https://mcmasteru365-my.sharepoint.com/:u:/g/personal/schankuc_mcmaster_ca/EbVY-gZQL-ZNuGsMRRAHCI8B-TNBKELD3HWeDsJCtq3oeA?e=0ymSJj), Password: DGMS-747

## 2. Example of usage

- **Step 1**: All hyper-parameters of our model can be found and customized in the `config.py` file. For instance, the employed graph neural network, the semantic matching 
  operation, the aggregation function, number of training iterations, etc.

- **Step2**: Examples of training & testing scripts are given as follows.
```shell
# java
python main.py --conv='rgcn' --filters='100' --match='submul' --match_agg='fc_max' --margin=0.5 --max_iter=216259 --val_start=100000 --valid_interval=10000 --log_dir='../JavaLogs/' --data_dir='../Datasets/java/'
# python
python main.py --conv='rgcn' --filters='100' --match='submul' --match_agg='fc_max' --margin=0.5 --max_iter=312189 --val_start=150000 --valid_interval=15000 --log_dir='../PythonLogs/' --data_dir='../Datasets/python/'
```

### Example Usage: Schankula & Li Data Preprocessing

1. **Step 1**: Clone the repo into the `$DGMS_ROOT/DGMS/` directory (we'll refer to
   this directory as the root repo directory).
2. **Step 2**: Download the [Stanford Parser](https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip) to parse the text snippets. 
Unzip the file and name the new directory as(`$DGMS_ROOT/stanford-parser-full-2020-11-17`) into the same
directory as the root repo directory. Move all jar files to the a new 
directory into `$DGMS_ROOT/stanford-parser-full-2020-11-17/jars/`. Unzip the models
jar file and place the English one in `$DGMS_ROOT/stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz`.
3. **Step 3**: Download the [GLoVe 840B-300d embeddings](https://nlp.stanford.edu/data/glove.840B.300d.zip) and place the `glove.840B.300d.txt` file in the `$DGMS_ROOT` directory.
4. Download the [CSN Python dataset](https://github.com/github/CodeSearchNet#downloading-data-from-s3) and place it in the `$DGMS_ROOT/python` directory.
5. Run `cd $DGMS_ROOT/DGMS/src` and then run `python DataPrep/ProcessCSN.py ` to preprocess the dataset.
6. The new dataset will be placed in `$DGMS_ROOT/python/final/processed` 
  directory.
