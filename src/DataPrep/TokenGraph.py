import DataPrep.LoadGLoVe as LoadGLoVe
import networkx as nx
from DataPrep.Constants import *
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

class TokenGraph:
    def __init__(self):
        self.G = nx.DiGraph()   # Graph of the sentence
        self.mapping = {}       # Mapping from node index to token
        self.i = 0              # Index of the next node to be added
        self.syntax_tokens = [] # List of tokens that existing in the initial code / sentence
        self.name_tokens = []


    def add_node(self, token, i = None, is_syntax_token = False, is_name_token = False):
        # Add a node to the graph
        if i is None:
            i = self.i
            self.i += 1

        self.mapping[i] = token

        # Add the token to list of syntax tokens if it is a syntax token
        if is_syntax_token:
            if isinstance(token, str):
                self.syntax_tokens.append(i)
            else:
                raise ValueError("Token is not a string")
            
        # Add the token to list of syntax tokens if it is a syntax token
        if is_name_token:
            if isinstance(token, str):
                self.name_tokens.append(i)
            else:
                raise ValueError("Token is not a string")

        # Add the node to the graph
        self.G.add_node(i)

        # Return the index of the new node
        return i
    
    def add_edge(self, parent, child = None, edge_attr = constituency):
        # Add an edge to the graph
        if child is None:
            child = self.i
            self.i += 1

        self.G.add_edge(parent, child, edge_attr = edge_attr)

    def add_word_ordering_edges(self, next_word = True, previous_word = True):
        # Add edges for word ordering
        for i in range(len(self.syntax_tokens) - 1):
            if next_word:
                self.add_edge(self.syntax_tokens[i],self.syntax_tokens[i+1], edge_attr=word_ordering)
            if previous_word:
                self.add_edge(self.syntax_tokens[i+1],self.syntax_tokens[i], edge_attr=word_ordering)

    def add_last_lexical_use_edges(self):
        # Add edges for last lexical use
        last_lexical_use_map = {}
        for i in range(len(self.name_tokens)):
            node_id = self.syntax_tokens[i]
            token = self.mapping[node_id]
            if token in last_lexical_use_map:
                last_use_node_id = last_lexical_use_map[token]
                self.add_edge(node_id, last_use_node_id, edge_attr=last_lexical_use)
                #print("Adding edge from ", node_id, "to", last_use_node_id)
            last_lexical_use_map[token] = node_id

    def list_nodes(self):
        num_nodes = self.G.number_of_nodes()

        nodes = []
        for i in range(num_nodes):
            nodes.append(self.mapping[i])
        
        return nodes

    def convert_to_pyg(self, glove):
        # Convert the networkx graph to a pytorch geometric graph
        edge_index = torch.tensor(list(self.G.edges)).t().contiguous()
        edge_attr = torch.tensor(np.array([self.G.edges[e]['edge_attr'] for e in self.G.edges]))
        x = np.empty((len(self.mapping), 300))
        for i in range(len(self.mapping)):
            x[i] = LoadGLoVe.word_to_glove(glove, self.mapping[i])
            #print(i, mapping[i], x[i])
        x = torch.tensor(x)

        return Data(x=x.float(), edge_index=edge_index, edge_attr=edge_attr.float())
    
    def save_graph(self, file_name = "graph.png", show_labels = True):
        graph_for_pos = self.G.copy()

        # Remove word-ordering edges
        for fr, to, dict in list(graph_for_pos.edges(data=True)):
            if np.array_equal(dict['edge_attr'],word_ordering):
                graph_for_pos.remove_edge(fr,to)

        for layer, nodes in enumerate(nx.topological_generations(graph_for_pos)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                graph_for_pos.nodes[node]["layer"] = layer
        
        graph = self.G.copy()

        pos = nx.multipartite_layout(graph_for_pos, subset_key="layer", align="horizontal")

        # Invert the positions on the y axis
        pos = {k: (-v[0], -v[1]) for k, v in pos.items()}

        #pos = nx.nx_pydot.graphviz_layout(self.G)

        # Create a new figure
        plt.figure()

        nx.draw(graph, pos=pos, with_labels=not show_labels, node_size=3000)

        edge_labels = {}
        for fr, to, dict in list(graph.edges(data=True)):
            if np.array_equal(dict['edge_attr'],word_ordering):
                edge_labels[(fr,to)] = "WO"
            elif np.array_equal(dict['edge_attr'],last_lexical_use):
                edge_labels[(fr,to)] = "LLU"
            elif np.array_equal(dict['edge_attr'],constituency):
                edge_labels[(fr,to)] = "CON"

        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=18)
        if show_labels: nx.draw_networkx_labels(graph, pos=pos, labels=self.mapping, font_size=18)
        
        plt.gcf().set_size_inches(12, 10)
        plt.savefig(file_name, dpi=300)

    def num_nodes(self):
        return self.G.number_of_nodes()
