import numpy as np
import sys
import pickle
import os
from DataPrep.WordSplit import try_glove_split

class GLoVe:

    def __init__(self, glove_file):
        self.word_to_vector = self.load_glove_vectors(glove_file)


    def load_glove_vectors(self, glove_file):
        pickle_file = os.path.splitext(glove_file)[0] + '.pkl'
        try:
            with open(pickle_file, 'rb') as f:
                word_to_vector = pickle.load(f)
        except FileNotFoundError:
            with open(glove_file, 'r', encoding='utf-8') as f:
                word_to_vector = {}
                i = 0
                for line in f:
                    i+=1
                    print("Processing line", i)
                    line = line.strip().split()
                    word = line[0]
                    try:
                        vector = np.array([float(x) for x in line[1:]])
                    except:
                        print("Error in line", line)
                    word_to_vector[word] = vector
            with open(pickle_file, 'wb') as f:
                pickle.dump(word_to_vector, f)
        return word_to_vector

    def get_vector(self, word):
        return self.word_to_vector.get(word, None)


def word_to_glove(glove, word):
    if glove.get_vector(word) is not None:
        return glove.get_vector(word)
    elif glove.get_vector(word.lower()) is not None:
        return glove.get_vector(word.lower())
    else:
        return try_glove_split(word, glove)


if __name__ == "__main__":
    glove_file = sys.argv[1]
    print("Loading GLoVe dataset...")
    glove = GLoVe(glove_file)

    while True:
        word = input("What word would you like to look up? ")
        vector = glove.get_vector(word)
        print(vector)