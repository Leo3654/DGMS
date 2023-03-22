import os
from nltk.parse import stanford
import networkx as nx

from nltk.tree import Tree
import pprint

import torch
from torch_geometric.utils.convert import from_networkx

from LoadGLoVe import GLoVe

import numpy as np

from WordSplit import *

os.environ['STANFORD_PARSER'] = '../../stanford-parser-full-2020-11-17/jars'
os.environ['STANFORD_MODELS'] = '../../stanford-parser-full-2020-11-17/jars'

constituency = np.array([1,0,0])
word_ordering = np.array([0,1,0])

# Initialize the GLoVe dataset
glove = GLoVe("../../glove.840B.300d.txt")


def sentence_to_pyg(text_to_parse):
    i = 0
    mapping = {}
    words = []

    def nltk_tree_to_graph(nltk_tree,i):
        """
        Converts an nltk tree to an nx graph.
        """
        nx_graph = nx.DiGraph()
        parent = i
        mapping[parent] = nltk_tree.label()
        for node in nltk_tree:
            if isinstance(node, Tree):
                print("Adding node ", i+1, " : ", node.label())
                i = i + 1
                nx_graph.add_edge(parent, i, edge_attr = constituency)
                new_graph, new_i = nltk_tree_to_graph(node,i)
                i = new_i
                nx_graph = nx.compose(nx_graph, new_graph)
            else:
                print("else", i+1, node)
                i=i + 1
                nx_graph.add_edge(parent,i, edge_attr = constituency)
                mapping[i] = node
                words.append(i)

        return (nx_graph, i)

    parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

    sentence = list(parser.raw_parse(text_to_parse))[0]

    graph, _ = nltk_tree_to_graph(sentence, 0)

    # Add word-ordering edges
    for i in range(len(words)-1):
        graph.add_edge(words[i], words[i+1], edge_attr = word_ordering)
        graph.add_edge(words[i+1], words[i], edge_attr = word_ordering)

    # lookup words' GLoVe vectors
    x = np.zeros((graph.number_of_nodes(), 300))

    for i in range(len(mapping)):
        if glove.get_vector(mapping[i]) is not None:
            x[i] = glove.get_vector(mapping[i])
        elif glove.get_vector(mapping[i].lower()) is not None:
            x[i] = glove.get_vector(mapping[i].lower())
        else:
            x[i] = try_glove_split(mapping[i], glove)

    #relabledGraph = nx.relabel_nodes(graph, mapping)
    #dict_repr = nx.to_dict_of_dicts(relabledGraph)
    #print(dict_repr)
    #print(nx.adjacency_matrix(graph))
    #print("Relabled:")
    #print(nx.adjacency_matrix(relabledGraph))
    #print(mapping)

    data = from_networkx(graph)
    data.x = torch.tensor(x)

    return data


if __name__ == "__main__":
    text_to_parse = input("Please enter a sentence: ")
    data = sentence_to_pyg(text_to_parse)

    print(data)
    print(data.x)