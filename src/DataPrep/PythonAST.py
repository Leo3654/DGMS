import ast
import networkx as nx
import matplotlib.pyplot as plt

from Constants import *

labels = {}
words = []

log = True

def add_new_node(G,i):
    G.add_node(i)
    if log:
        print("Adding node", i)

def add_new_edge(G,i,j):
    G.add_edge(i,j)
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
    else:
        i = i + 1
        add_new_edge(G,parent,i)
        labels[i] = str(const)
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
        add_new_edge(G, i, i + 1)
        labels[i+1] = node.id
        words.append(i)
        i = i + 1
    elif isinstance(node, ast.Store):
        add_new_node(G,i)
        labels[i] = "="
        words.append(i)
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
tree = ast.parse(code3)

print(ast.dump(tree))

graph = ast_to_networkx(tree)


for layer, nodes in enumerate(nx.topological_generations(graph)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
    for node in nodes:
        graph.nodes[node]["layer"] = layer

print(labels)
print(words)

pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
show_labels = True

nx.draw(graph, pos=pos, with_labels=not show_labels)
if show_labels: nx.draw_networkx_labels(graph, pos=pos, labels=labels)
plt.show()