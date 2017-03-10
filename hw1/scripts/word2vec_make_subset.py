#!/usr/bin/python3
"""
This script create a subset of word2vec file.

Usage:
    python3 word2vec_make_subset.py <word2vec> <dict.json> <out-file>

The original word2vec is a huge file, making it diffcult to deploy and run.
Therefore, we create a binary file only including a subset of word2vec, which
is the intersaction of words that appears in the corpus and the original
'GoogleNews-vector-negative300.bin'
"""

import os
import sys
import json
import struct
import numpy as np

# Required to import word2vec
import __init__
import word2vec

def main(argv):
    model = word2vec.load_word2vec_model(argv[1])
    word_dict = json.load(open(argv[2]))

    output = {}
    for word, index in word_dict.items():
        if isinstance(word, int):
            print(word)
            continue
        if index >= 30000 or word not in model:
            continue
        output[index] = model[word]

    output[1] = np.zeros((300,))    # ','

    word2vec.dump_indexed_model(argv[3], output)
    print('Successfully write to {0}'.format(argv[3]))


def print_usage():
    print('Usage:')
    print('\t{0}, <word2vec> <dict.json> <out-file>'.format(
        os.path.basename(__file__)
    ))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print_usage()
    else:
        main(sys.argv)
else:
    print_usage()
