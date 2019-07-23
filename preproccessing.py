import nltk
import numpy as np
import random
import pickle

from collections import Counter

def create_lexicon(pos, neg, num_of_lines=10000000, min_occurrences = 50, max_occurrences=1000):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    words = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:num_of_lines]:
                words += list(nltk.tokenizer.word_tokenizer(l))

    words = [lemmatizer.lemmatize(word) for word in words]
    word_counts = Counter(words)
    
    lexicon = []
    for w in word_counts:
        if max_occurrences > w > min_occurrences:
            lexicon.append(w)

    return lexicon