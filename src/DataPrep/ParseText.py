import os
from nltk.parse import stanford
import networkx as nx

from nltk.tree import Tree

os.environ['STANFORD_PARSER'] = '../../stanford-parser-full-2020-11-17/jars'
os.environ['STANFORD_MODELS'] = '../../stanford-parser-full-2020-11-17/jars'

i = 0
mapping = {}

def nltk_tree_to_graph(nltk_tree):
    """
    Converts an nltk tree to an nx graph.
    """
    global i, label
    nx_graph = nx.DiGraph()
    for node in nltk_tree:
        parent = i
        if isinstance(node, Tree):
            print("Adding node ", i, " : ", node.label())
            mapping[parent] = node.label()
            i = i + 1
            nx_graph.add_edge(parent, i)
            nx_graph = nx.compose(nx_graph, nltk_tree_to_graph(node))
        else:
            print("else", node)
            mapping[parent] = node
            i=i+1
    return nx_graph

parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sentences = list(parser.raw_parse("Hello, My name is Melroy."))[0]
print("Sentences:", sentences)
trees = nltk_tree_to_graph(sentences)
relabledTrees = nx.relabel_nodes(trees, mapping)
print(nx.adjacency_matrix(relabledTrees))
print(mapping)

