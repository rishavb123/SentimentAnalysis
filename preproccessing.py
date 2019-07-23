import nltk
import numpy as np
import random
import pickle

from collections import Counter

lemmatizer = nltk.stem.WordNetLemmatizer()

def create_lexicon(pos, neg, num_of_lines=10000000, min_occurrences = 50, max_occurrences=1000):
    words = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:num_of_lines]:
                words += list(nltk.tokenize.word_tokenizer(l.lower()))

    words = [lemmatizer.lemmatize(word) for word in words]
    word_counts = Counter(words)
    
    lexicon = []
    for w in word_counts:
        if max_occurrences > w > min_occurrences:
            lexicon.append(w)

    return lexicon

def create_features(sample, lexicon, classification, num_of_lines=10000000):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:num_of_lines]:
            current_words = nltk.tokenize.word_tokenizer(l.lower())
            current_words = [lemmatizer.lemmatize(word) for word in current_words]
            features = [0 for _ in range(len(lexicon))]
            for word in current_words:
                if word.lower() in lexicon:
                    features[lexicon.index(word)] = 1
            features = list(features)
            featureset.append([features, classification])

    return featureset

def structure_data(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    
    features = create_features(pos, [1, 0]) + create_features(neg, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    
    testing_size = int(test_size * len(features))
    
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

def save_structured_data(pos, neg, data_file):
    train_x, train_y, test_x, test_y = structure_data(pos, neg)
    with open(data_file, 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)

def load_data(data_file):
    with open(data_file, 'rb') as f:
        [train_x, train_y, test_x, test_y] = pickle.load(f)
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    save_structured_data('./data/positive_strings.txt', './data/negative_strings.txt', './data/data.pkl')