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

# Convert a sentence to a PyG model
def sentence_to_pyg(text_to_parse):
    i = 0
    mapping = {}
    words = []

    # Convert the NLTK tree to a graph
    def nltk_tree_to_graph(nltk_tree,*args):
        """
        Converts an nltk tree to an nx graph.
        """

        # If i is not provided, use 0
        if len(args) == 0:
            i = 0
        else:
            i = args[0]
        nx_graph = nx.DiGraph()
        parent = i
        mapping[parent] = nltk_tree.label()

        # For all child nodes of this tree
        for child_node in nltk_tree:
            # if the node is a tree, process it recursively
            if isinstance(child_node, Tree):
                i = i + 1
                # Connect the parent node to the child node
                nx_graph.add_edge(parent, i, edge_attr = constituency)
                # Recursively process the child nodes
                new_graph, new_i = nltk_tree_to_graph(child_node, i)
                i = new_i
                # Compose the two graphs
                nx_graph = nx.compose(nx_graph, new_graph)
            else: # Node is a leaf node
                i = i + 1
                # Connect parent to leaf node
                nx_graph.add_edge(parent,i, edge_attr = constituency)
                # Store the string text
                mapping[i] = child_node
                # Keep track of the fact that this node is a leaf node
                words.append(i)

        # if an i was given, return it
        if len(args) == 0:
            return nx_graph
        elif len(args) == 1:
            return (nx_graph, i)
        else:
            return None

    # load the English Stanford parser
    parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

    # parse the sentence into an NLTK tree
    sentence = list(parser.raw_parse(text_to_parse))[0]

    # convert the NLTK tree to a graph
    graph = nltk_tree_to_graph(sentence)

    # Add word-ordering edges
    for i in range(len(words)-1):
        graph.add_edge(words[i], words[i+1], edge_attr = word_ordering)
        graph.add_edge(words[i+1], words[i], edge_attr = word_ordering)

    # Lookup words' GLoVe vectors and store them in the x vector
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

    # Convert the graph to a PyG object
    data = from_networkx(graph)

    # Attach the GLoVe vectors to the PyG object
    data.x = torch.tensor(x)

    # Return the PyG object
    return data


if __name__ == "__main__":
    text_to_parse = input("Please enter a sentence: ")
    data = sentence_to_pyg(text_to_parse)

    print(data)
    print(data.x)