import ast
import networkx as nx
import matplotlib.pyplot as plt

from Constants import *

import asttokens

labels = {}
words = []
last_use_map = {}

log = True

def add_new_node(G,i):
    G.add_node(i)
    if log:
        print("Adding node", i)

def add_new_edge(G,i,j, edge_attr = constituency):
    G.add_edge(i,j,edge_attr = edge_attr)
    if log:
        print("Adding edge", i, j)

def const_to_graph(const, parent):
    G = nx.DiGraph()
    i = parent
    if isinstance(const, str):
        for word in str.split(const):
            i = i + 1
            add_new_edge(G,parent,i)
            labels[i] = word
            words.append(i)
    else:
        i = i + 1
        add_new_edge(G,parent,i)
        labels[i] = str(const)
        words.append(i)
    return (G, i)


def ast_to_networkx(node, *args):
    global labels

    if len(args) == 0:
        i = 0
    else:
        i = args[0]

    G = nx.DiGraph()
    parent = i
    if isinstance(node, ast.Constant):
        # add "Constant" node
        add_new_node(G,i)

        labels[i] = "Constant"
        # add node with constant
        add_new_edge(G, i, i + 1)
        labels[i+1] = str(node.value)
        new_graph, i = const_to_graph(node.value, i)
        G = nx.compose(G, new_graph)
        i = i + 1
    elif isinstance(node, ast.Name):
        # add "Name" node
        add_new_node(G,i)
        labels[i] = "Name"
        # add node with constant
        add_new_edge(G, i, i + 1, edge_attr=constituency)
        labels[i+1] = node.id

        try:
            last_use = last_use_map[node.id]
            add_new_edge(G, i+1, last_use, last_lexical_use)
        except:
            pass

        last_use_map[node.id] = i + 1
        i = i + 1
        words.append(i)
    elif isinstance(node, ast.Store):
        add_new_node(G,i)
        labels[i] = "Store"
    elif isinstance(node, ast.AST):
        add_new_node(G,i)
        labels[i] = node.__class__.__name__
    else:
        add_new_node(G,i)
        labels[i] = str(node)
        words.append(i)

    i = i + 1

    for child in ast.iter_child_nodes(node):
        add_new_edge(G, parent, i)
        new_graph, new_i = ast_to_networkx(child, i)
        i = new_i
        G = nx.compose(G, new_graph)

        # if an i was given, return it
    if len(args) == 0:
        return G
    elif len(args) == 1:
        return (G, i)
    else:
        return None

code = "x = 1\ny = 2\nz = x + y * y"
code2 = "print(\"Hello world\", 5)"
code3 = "for i in range(4): print(\"Hello World {i}\")"
code4 = "with open('file_path', 'w') as file:\n    file.write('hello world !')"
tokens_graph = asttokens.ASTTokens(code4, parse=True)
tree = tokens_graph.tree

print(tokens_graph)

print(ast.dump(tree))

graph = ast_to_networkx(tree)

for node in ast.walk(tokens_graph.tree):
    if hasattr(node, 'lineno'):
        print("first token:", node.first_token)
        print(tokens_graph.get_text_range(node), node.__class__.__name__, tokens_graph.get_text(node))
        print("---------")

for layer, nodes in enumerate(nx.topological_generations(graph)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
    for node in nodes:
        graph.nodes[node]["layer"] = layer

print(labels)
for word in words: print(word, labels[word])

for i in range(len(words)-1):
     graph.add_edge(words[i],words[i+1], edge_attr=word_ordering)

pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")

for k in pos:
    pos[k][1] = -pos[k][1]

show_labels = True

nx.draw(graph, pos=pos, with_labels=not show_labels)

edge_labels = {}
for fr, to, dict in list(graph.edges(data=True)):
    edge_labels[(fr,to)] = dict['edge_attr']

nx.draw_networkx_edge_labels(graph, pos, edge_labels)
if show_labels: nx.draw_networkx_labels(graph, pos=pos, labels=labels)
plt.show()