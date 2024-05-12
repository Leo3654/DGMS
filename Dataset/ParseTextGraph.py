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

from nltk.tokenize import sent_tokenize

os.environ['STANFORD_PARSER'] = './stanford-parser-full-2020-11-17/jars'
os.environ['STANFORD_MODELS'] = './stanford-parser-full-2020-11-17/jars'

# Initialize the GLoVe dataset
glove = GLoVe("./glove.840B.300d.txt")


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

        parser = stanford.StanfordParser(model_path="./stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

        sentence = pre_process_docstring(text_to_parse)
        print("Sentence:", sentence)

        # parse the sentence into an NLTK tree
        tree = list(parser.raw_parse(sentence))[0]

        # convert the NLTK tree to a graph
        nltk_tree_to_graph(tree)

        # Convert the NLTK tree to a graph  

        # load the English Stanford parser
        # parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


        # # Convert the graph to a PyG object
        # data = from_networkx(graph)


        # # Attach the GLoVe vectors to the PyG object
        # data.x = torch.tensor(x)

        # # Return the PyG object
        # return data

def pre_process_docstring(docstring):
    # Split the docstring into sentences
    sentences = sent_tokenize(docstring)

    return sentences[0]

if __name__ == "__main__":
    text_to_parse = """Compute a histogram using the provided buckets. The buckets
        are all open to the right except for the last which is closed.
        e.g. [1,10,20,50] means the buckets are [1,10) [10,20) [20,50],
        which means 1<=x<10, 10<=x<20, 20<=x<=50. And on the input of 1
        and 50 we would have a histogram of 1,0,1.

        If your histogram is evenly spaced (e.g. [0, 10, 20, 30]),
        this can be switched from an O(log n) inseration to O(1) per
        element (where n is the number of buckets).

        Buckets must be sorted, not contain any duplicates, and have
        at least two elements.

        If `buckets` is a number, it will generate buckets which are
        evenly spaced between the minimum and maximum of the RDD. For
        example, if the min value is 0 and the max is 100, given `buckets`
        as 2, the resulting buckets will be [0,50) [50,100]. `buckets` must
        be at least 1. An exception is raised if the RDD contains infinity.
        If the elements in the RDD do not vary (max == min), a single bucket
        will be used.

        The return value is a tuple of buckets and histogram.""".replace("\n"," ")
    #input("Please enter a sentence: ")
    docstring_graph = DocstringGraph(text_to_parse)

    docstring_graph.add_word_ordering_edges()

    glove = GLoVe("/glove.840B.300d.txt")

    graph = docstring_graph.convert_to_pyg(glove)

    docstring_graph.save_graph()

    print("Number of nodes:", docstring_graph.num_nodes())