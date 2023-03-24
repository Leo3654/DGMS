from python_graphs import program_graph
from python_graphs import program_graph_graphviz
import sys
import ast

program = "x=1\ny=2\nz=x+y"

graph = program_graph.get_program_graph(program)

tree = ast.parse(program)


for node in ast.walk(tree):
    print(node)
    print(node.__dict__)
    #print("children: " + str([x for x in ast.iter_child_nodes(node)]) + "\\n")

print(len(graph.all_nodes()))

program_graph_graphviz.render(graph, "graph.png")