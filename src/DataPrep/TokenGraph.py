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


    def add_node(self, token, i = None, is_syntax_token = True):
        if i is None:
            i = self.i
            self.i += 1

        if is_syntax_token:
            self.syntax_tokens.append(i)

        self.G.add_node(i)

        return i
    
    def add_edge(self, parent, edge_attr = constituency, child = None):
        if child is None:
            child = self.i
            self.i += 1

        self.G.add_edge(parent, child, edge_attr = edge_attr)

    def add_word_ordering_edges(self, next_word = True, previous_word=True):
        for i in range(len(self.syntax_tokens) - 1):
            if next_word:
                self.add_edge(i, word_ordering, i+1)
            if previous_word:
                self.add_edge(i+1, word_ordering, i)

    def convert_to_pyg(self, glove):
        # Convert the networkx graph to a pytorch geometric graph
        edge_index = torch.tensor(list(self.G.edges)).t().contiguous()
        edge_attr = torch.tensor([self.G.edges[e]['edge_attr'] for e in self.G.edges])
        x = np.empty((len(self.mapping), 300)
        for i in range(len(self.mapping)):
            if glove.get_vector(self.mapping[i]) is not None:
                x[i] = glove.get_vector(self.mapping[i])
            elif glove.get_vector(self.mapping[i].lower()) is not None:
                x[i] = glove.get_vector(self.mapping[i].lower())
            else:
                x[i] = try_glove_split(mapping[i], glove)
            #print(i, mapping[i], x[i])
        x = torch.tensor(x)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    

    