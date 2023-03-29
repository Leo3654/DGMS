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

from Constants import constituency, word_ordering

from TokenGraph import TokenGraph

os.environ['STANFORD_PARSER'] = '../../stanford-parser-full-2020-11-17/jars'
os.environ['STANFORD_MODELS'] = '../../stanford-parser-full-2020-11-17/jars'

# Initialize the GLoVe dataset
glove = GLoVe("../../glove.840B.300d.txt")


class DocstringGraph(TokenGraph):
    def __init__(self, docstring):
        super().__init__()
        self.sentence_to_networkx(docstring)

    # Convert a sentence to a PyG model
    def sentence_to_networkx(self,text_to_parse):
        # Convert the NLTK tree to a graph
        def nltk_tree_to_graph(node):
            """
            Converts an nltk tree to an nx graph.
            """

            parent_node_id = self.add_node(node.label())

            # For all child nodes of this tree
            for child_node in node:
                # if the node is a tree, process it recursively
                if isinstance(child_node, Tree):
                    # Recursively process the child nodes
                    child_node_id = nltk_tree_to_graph(child_node)
                    self.add_edge(parent_node_id, child_node_id, edge_attr = constituency)
                else: # Node is a leaf node
                    child_node_id = self.add_node(child_node, is_syntax_token = True)
                    self.add_edge(parent_node_id, child_node_id, edge_attr = constituency)
            
            return parent_node_id

        parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

        # parse the sentence into an NLTK tree
        sentence = list(parser.raw_parse(text_to_parse))[0]

        # convert the NLTK tree to a graph
        graph = nltk_tree_to_graph(sentence)

        # Convert the NLTK tree to a graph  

        # load the English Stanford parser
        # parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


        # # Convert the graph to a PyG object
        # data = from_networkx(graph)


        # # Attach the GLoVe vectors to the PyG object
        # data.x = torch.tensor(x)

        # # Return the PyG object
        # return data


if __name__ == "__main__":
    text_to_parse = input("Please enter a sentence: ")
    docstring_graph = DocstringGraph(text_to_parse)

    docstring_graph.add_word_ordering_edges()

    glove = GLoVe("../../glove.840B.300d.txt")

    graph = docstring_graph.convert_to_pyg(glove)

    docstring_graph.save_graph()