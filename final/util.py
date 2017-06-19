import pickle
import numpy as np

def load_worddict(filepath, vocab_size):
    with open(filepath, 'rb') as f:
        wordcount = pickle.load(f)[0]

    w2id_dict = {}
    for i, (word, _) in enumerate(wordcount):
        w2id_dict[word] = i
    
        if i == vocab_size - 1:
            return w2id_dict