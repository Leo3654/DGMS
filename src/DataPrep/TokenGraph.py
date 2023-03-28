import LoadGLoVe
import networkx as nx
from Constants import *
import torch
import matplotlib.pyplot as plt

class TokenGraph:
    def __init__(self):
        self.G = nx.DiGraph()   # Graph of the sentence
        self.mapping = {}       # Mapping from node index to token
        self.i = 0              # Index of the next node to be added
        self.syntax_tokens = [] # List of tokens that existing in the initial code / sentence


    def add_node(self, token, i = None, is_syntax_token = False):
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
        for i in range(len(self.syntax_tokens) - 1):
            token = self.syntax_tokens[i]
            if token in last_lexical_use_map:
                self.add_edge(self.syntax_tokens[i], last_lexical_use, last_lexical_use_map[token])
            last_lexical_use_map[token] = self.syntax_tokens[i]

    def convert_to_pyg(self, glove):
        # Convert the networkx graph to a pytorch geometric graph
        edge_index = torch.tensor(list(self.G.edges)).t().contiguous()
        edge_attr = torch.tensor([self.G.edges[e]['edge_attr'] for e in self.G.edges])
        x = np.empty((len(self.mapping), 300))
        for i in range(len(self.mapping)):
            x[i] = LoadGLoVe.word_to_glove(glove, self.mapping[i])
            #print(i, mapping[i], x[i])
        x = torch.tensor(x)

        return torch.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def save_graph(self, show_labels = True):
        graph = self.G.copy()

        for layer, nodes in enumerate(nx.topological_generations(graph)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
            for node in nodes:
                graph.nodes[node]["layer"] = layer

        #pos = nx.multipartite_layout(graph, subset_key="layer", align="horizontal")
        pos = nx.nx_pydot.graphviz_layout(G)

        nx.draw(graph, pos=pos, with_labels=not show_labels)

        edge_labels = {}
        for fr, to, dict in list(graph.edges(data=True)):
            edge_labels[(fr,to)] = dict['edge_attr']

        nx.draw_networkx_edge_labels(graph, pos, edge_labels)
        if show_labels: nx.draw_networkx_labels(graph, pos=pos, labels=self.mapping)
        
        plt.gcf().set_size_inches(10, 8)
        plt.savefig("graph.png", dpi=300)

    