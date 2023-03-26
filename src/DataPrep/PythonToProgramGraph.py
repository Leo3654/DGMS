from python_graphs import program_graph
from python_graphs import program_graph_graphviz
import sys
import ast
from PythonUtils import get_reduced_program_graph, render

program = "x=1"

graph = get_reduced_program_graph(program)

tree = ast.parse(program)

print(ast.dump(tree))

print(graph.dump_tree())

for node in ast.walk(tree):
    print(node)
    print(node.__dict__)
    #print("children: " + str([x for x in ast.iter_child_nodes(node)]) + "\\n")

# print(len(graph.all_nodes()))

render(graph, "graph.png")