import os
from nltk.parse import stanford
import networkx as nx
os.environ['STANFORD_PARSER'] = '../../stanford-parser-full-2020-11-17/jars'
os.environ['STANFORD_MODELS'] = '../../stanford-parser-full-2020-11-17/jars'

def nltk_tree_to_graph(nltk_tree):
    """
    Converts an nltk tree to an nx graph.
    """
    nx_graph = nx.Graph()
    for node in nltk_tree:
        if isinstance(node, Tree):
            nx_graph.add_edge(node.label(), node[0])
            nx_graph = nx.compose(nx_graph, nltk_tree_to_graph(node))
        else:
            nx_graph.add_node(node)
    return nx_graph

parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sentences = parser.raw_parse("Hello, My name is Melroy.")
print(nltk_tree_to_graph(list(sentences)[0]))

