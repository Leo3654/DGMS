import numpy as np
import re

def camel_case_split(identifier):
    # Use regular expressions to split the identifier into words
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    # Return a list of the words
    return [m.group(0) for m in matches]

def snake_case_split(identifier):
    # Split the identifier into words using underscores
    return identifier.split('_')

def try_glove_split(word, glove):
    # Split the word into multiple words using camel case and snake case
    camel_split = [glove.get_vector(individual_word) for individual_word in camel_case_split(word)]
    # If all of the word vectors are present, return the mean of the word vectors
    if not any(v is None for v in camel_split):
        return np.mean(camel_split, axis=0)
    
    # Otherwise, split the word into multiple words using snake case
    snake_split = [glove.get_vector(individual_word) for individual_word in snake_case_split(word)]
    # If all of the word vectors are present, return the mean of the word vectors
    if not any(v is None for v in camel_split):
        return np.mean(snake_split, axis=0)
    
    # Otherwise, return a zero vector
    return np.zeros(300)