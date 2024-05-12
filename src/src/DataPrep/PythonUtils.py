from python_graphs import program_utils
from python_graphs.program_graph import *
import gast as ast
import pygraphviz

def get_reduced_program_graph(program):
  """Constructs a program graph to represent the given program."""
  program_node = program_utils.program_to_ast(program)  # An AST node.

  # TODO(dbieber): Refactor sections of graph building into separate functions.
  program_graph = ProgramGraph()

#   # Perform control flow analysis.
#   control_flow_graph = control_flow.get_control_flow_graph(program_node)

#   # Add AST_NODE program graph nodes corresponding to Instructions in the
#   # control flow graph.
#   for control_flow_node in control_flow_graph.get_control_flow_nodes():
#     program_graph.add_node_from_instruction(control_flow_node.instruction)

  # Add AST_NODE program graph nodes corresponding to AST nodes.
  for ast_node in ast.walk(program_node):
    if not program_graph.contains_ast_node(ast_node):
      pg_node = make_node_from_ast_node(ast_node)
      program_graph.add_node(pg_node)

  root = program_graph.get_node_by_ast_node(program_node)
  program_graph.root_id = root.id

  # Add AST edges (FIELD). Also add AST_LIST and AST_VALUE program graph nodes.
  for ast_node in ast.walk(program_node):
    for field_name, value in ast.iter_fields(ast_node):
      if isinstance(value, list):
        pg_node = make_node_for_ast_list()
        program_graph.add_node(pg_node)
        program_graph.add_new_edge(
            ast_node, pg_node, pb.EdgeType.FIELD, field_name)
        for index, item in enumerate(value):
          list_field_name = make_list_field_name(field_name, index)
          if isinstance(item, ast.AST):
            program_graph.add_new_edge(pg_node, item, pb.EdgeType.FIELD,
                                       list_field_name)
          else:
            item_node = make_node_from_ast_value(item)
            program_graph.add_node(item_node)
            program_graph.add_new_edge(pg_node, item_node, pb.EdgeType.FIELD,
                                       list_field_name)
      elif isinstance(value, ast.AST):
        program_graph.add_new_edge(
            ast_node, value, pb.EdgeType.FIELD, field_name)
      else:
        pg_node = make_node_from_ast_value(value)
        program_graph.add_node(pg_node)
        program_graph.add_new_edge(
            ast_node, pg_node, pb.EdgeType.FIELD, field_name)

  # Add SYNTAX_NODE nodes. Also add NEXT_SYNTAX and LAST_LEXICAL_USE edges.
  # Add these edges using a custom AST unparser to visit leaf nodes in preorder.
  SyntaxNodeUnparser(program_node, program_graph)

  return program_graph


def to_graphviz(graph):
  """Creates a graphviz representation of a ProgramGraph.
  Args:
    graph: A ProgramGraph object to visualize.
  Returns:
    A pygraphviz object representing the ProgramGraph.
  """
  g = pygraphviz.AGraph(strict=False, directed=True)
  print(graph.nodes)
  for unused_key, node in graph.nodes.items():
    node_attrs = {}
    if node.ast_type:
      node_attrs['label'] = six.ensure_str(node.ast_type, 'utf-8')
    else:
      node_attrs['label'] = str(graph.nodes[node])
    node_type_colors = {
    }
    if node.node_type in node_type_colors:
      node_attrs['color'] = node_type_colors[node.node_type]
      node_attrs['colorscheme'] = 'svg'

    g.add_node(node.id, **node_attrs)
  for edge in graph.edges:
    edge_attrs = {}
    edge_attrs['label'] = edge.type.name
    edge_colors = {
        pb.EdgeType.LAST_READ: 'red',
        pb.EdgeType.LAST_WRITE: 'red',
    }
    if edge.type in edge_colors:
      edge_attrs['color'] = edge_colors[edge.type]
      edge_attrs['colorscheme'] = 'svg'
    g.add_edge(edge.id1, edge.id2, **edge_attrs)
  return g


def render(graph, path='/tmp/graph.png'):
  g = to_graphviz(graph)
  g.draw(path, prog='dot')