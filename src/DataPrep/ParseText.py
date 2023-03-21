import os
from nltk.parse import stanford
import networkx as nx

from nltk.tree import Tree
import pprint

from torch_geometric.utils.convert import from_networkx

from LoadGLoVe import GLoVe


os.environ['STANFORD_PARSER'] = '../../stanford-parser-full-2020-11-17/jars'
os.environ['STANFORD_MODELS'] = '../../stanford-parser-full-2020-11-17/jars'

i = 0
mapping = {}
words = []

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
            print("Adding node ", i+1, " : ", node.label())
            i = i + 1
            nx_graph.add_edge(parent, i, type = "constituency")
            nx_graph = nx.compose(nx_graph, nltk_tree_to_graph(node))
        else:
            print("else", i+1, node)
            i=i + 1
            nx_graph.add_edge(parent,i)
            mapping[i] = node
            words.append(i)

    return nx_graph


parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
text_to_parse = input("Please enter a sentence: ")
sentences = list(parser.raw_parse(text_to_parse))[0]
print("Sentences:", sentences)
graph = nltk_tree_to_graph(sentences)

# Add word-ordering edges
for i in range(len(words)-1):
    graph.add_edge(words[i], words[i+1], type="word_ordering")
    graph.add_edge(words[i+1], words[i], type="word_ordering")

# convert words to GLoVe vectors

glove = GLoVe("../../glove.840B.300d.txt")

mapping = {key: glove.get_vector(word) for key, word in mapping.items()}

relabledGraph = nx.relabel_nodes(graph, mapping)
dict_repr = nx.to_dict_of_dicts(relabledGraph)
print(dict_repr)
print(nx.adjacency_matrix(graph))
print("Relabled:")
print(nx.adjacency_matrix(relabledGraph))
print(mapping)

data = from_networkx(graph)

print(data)