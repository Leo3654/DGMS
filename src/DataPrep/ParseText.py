import os
from nltk.parse import stanford
import networkx as nx

from nltk.tree import Tree
import pprint

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
    parent = i
    mapping[parent] = nltk_tree.label()
    for node in nltk_tree:
        if isinstance(node, Tree):
            print("Adding node ", i, " : ", node.label())
            i = i + 1
            nx_graph.add_edge(parent, i)
            nx_graph = nx.compose(nx_graph, nltk_tree_to_graph(node))
        else:
            print("else", i, node)
            i=i + 1
            mapping[i] = node
    return nx_graph


parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
text_to_parse = input("Please enter a sentence: ")
sentences = list(parser.raw_parse(text_to_parse))[0]
print("Sentences:", sentences)
trees = nltk_tree_to_graph(sentences)
relabledTrees = nx.relabel_nodes(trees, mapping)
dict_repr = nx.to_dict_of_dicts(relabledTrees)
print(pprint.pprint(dict_repr, indent=1, width=20))
print(mapping)

