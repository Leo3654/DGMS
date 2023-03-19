import numpy as np
import sys

def load_glove_vectors(glove_file):
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
    return word_to_vector

glove_file = sys.argv[1]
print("Loading GLoVe dataset...")
word_to_vector = load_glove_vectors(glove_file)

def get_vector(word):
    return word_to_vector.get(word, None)

while True:
    word = input("What word would you like to look up?")
    vector = get_vector(word)
    print(vector)