import numpy as np

def load_glove_vectors(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        word_to_vector = {}
        for line in f:
            line = line.strip().split()
            word = line[0]
            vector = np.array([float(x) for x in line[1:]])
            word_to_vector[word] = vector
    return word_to_vector

glove_file = 'glove.840B.300d.txt'
word_to_vector = load_glove_vectors(glove_file)

def get_vector(word):
    return word_to_vector.get(word, None)

while True:
    word = input("What word would you like to look up?")
    vector = get_vector(word)
    print(vector)