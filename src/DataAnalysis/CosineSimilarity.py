from DataPrep.TokenGraph import TokenGraph
from DataPrep.LoadGLoVe import GLoVe
from DataPrep.ParsePythonGraph import PythonGraph
from DataPrep.ParseTextGraph import TextGraph

glove = GLoVe("../../glove.840B.300d.txt")

code = """print("Hello World")"""
text = "Prints Hello World"

code_graph = PythonGraph(code)
text_graph = TextGraph(text)

code_graph_pyg = code_graph.convert_to_pyg()
text_graph_pyg = text_graph.convert_to_pyg()

code_graph_glove = code_graph_pyg.x
text_graph_glove = text_graph_pyg.x

#Compute the cosine similarity between the two graphs
