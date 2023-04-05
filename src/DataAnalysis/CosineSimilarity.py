from DataPrep.TokenGraph import TokenGraph
from DataPrep.LoadGLoVe import GLoVe
from DataPrep.ParsePythonGraph import PythonGraph
from DataPrep.ParseTextGraph import DocstringGraph

import numpy as np

glove = GLoVe("../../glove.840B.300d.txt")

code = """print("Hello World")"""
text = "prints Hello World"

code_graph = PythonGraph(code)
text_graph = DocstringGraph(text)

code_graph_pyg = code_graph.convert_to_pyg(glove)
text_graph_pyg = text_graph.convert_to_pyg(glove)

code_graph_glove = code_graph_pyg.x
text_graph_glove = text_graph_pyg.x

#Compute the cosine similarity between two vectors
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#Compute the cosine similarity between two graphs
def cosine_similarity_graphs(g1, g2):
    g1_nodes = g1.x
    g2_nodes = g2.x

    similarity_matrix = np.empty((g1_nodes.shape[0], g2_nodes.shape[0]))

    for i in range(g1_nodes.shape[0]):
        for j in range(g2_nodes.shape[0]):
            similarity_matrix[i][j] = cosine_similarity(g1_nodes[i], g2_nodes[j])

    return similarity_matrix


cosine_similarity_matrix = cosine_similarity_graphs(code_graph_pyg, text_graph_pyg)

code_graph_nodes = code_graph.list_nodes()
text_graph_nodes = text_graph.list_nodes()

print(",", end="")

for text in text_graph_nodes:
    print(text, ",", end="")

print("")

for i in range(cosine_similarity_matrix.shape[0]):
    print(code_graph_nodes[i], ",", end="")
    for j in range(cosine_similarity_matrix.shape[1]):
        print(cosine_similarity_matrix[i][j], ",", end="")
    print("")

#print(cosine_similarity_graphs(code_graph_pyg, text_graph_pyg))

text_graph.add_word_ordering_edges()

code_graph.save_graph("code_graph.pdf")
text_graph.save_graph("text_graph.pdf")