import os
from nltk.parse import stanford
os.environ['STANFORD_PARSER'] = '../../stanford-parser-full-2020-11-17/jars'
os.environ['STANFORD_MODELS'] = '../../stanford-parser-full-2020-11-17/jars'

parser = stanford.StanfordParser(model_path="../../stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sentences = parser.raw_parse("Hello, My name is Melroy.")
print(nltk_tree_to_graph(sentences))

def nltk_tree_to_graph(nltk_tree):
    graph = nx.Graph()
    labels = {}

    def traverse_tree(node, parent=None):
        if isinstance(node, str):
            node = Tree(node, [])
        labels[node] = node.label()

        if parent is not None:
            graph.add_edge(parent, node)

        for child in node:
            if isinstance(child, str):
                child = Tree(child, [])
            traverse_tree(child, node)

    traverse_tree(nltk_tree)
    graph = nx.relabel_nodes(graph, labels)
    return graph