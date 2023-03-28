import LoadGLoVe
import networkx as nx
from Constants import *
import torch

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
    
    def add_edge(self, parent, edge_attr = constituency, child = None):
        # Add an edge to the graph
        if child is None:
            child = self.i
            self.i += 1

        self.G.add_edge(parent, child, edge_attr = edge_attr)

    def add_word_ordering_edges(self, next_word = True, previous_word = True):
        # Add edges for word ordering
        for i in range(len(self.syntax_tokens) - 1):
            if next_word:
                self.add_edge(i, word_ordering, i+1)
            if previous_word:
                self.add_edge(i+1, word_ordering, i)

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
        x = np.empty((len(self.mapping), 300)
        for i in range(len(self.mapping)):
            x[i] = word_to_glove(glove, self.mapping[i])
            #print(i, mapping[i], x[i])
        x = torch.tensor(x)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    

    