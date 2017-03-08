# This module is copied and modified from gensim
# The original code can be mostly found on
# https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py

"""
A module to load pretrained Word2Vec model
Example usage:
>>> import Word2Vec

>>> model = Word2Vec.load_model('../data/word2vec/GoogleNews-vectors-negative300.bin')
loading projection weights from ../data/word2vec/GoogleNews-vectors-negative300.bin
loaded (3000000, 300) matrix from ../data/word2vec/GoogleNews-vectors-negative300.bin

>>> model['the'].shape
(1, 300)
"""

import numpy as np
import os


__all__ = [
    'Word2Vec',
    'load_model'
]


def smart_open(fname, mode='rb'):
    """
    Open file based on file extension
    It may be required to import gzip or bz2 module.
    """
    _, ext = os.path.splitext(fname)
    if ext == '.bz2':
        from bz2 import BZ2File
        return BZ2File(fname, mode)
    if ext == '.gz':
        from gzip import GzipFile
        return GzipFile(fname, mode)
    return open(fname, mode)


class Word2Vec():
    """
    Word2Vec model
    Usage:
        Use this model as a dict. It accept a single word or a list of words as
        key.
    Example::
    # Return array with shape (1, word_vec_len)
    >>> trained_model['office']
    array([[ -1.40128313e-02, ...]])

    # Return array with shape (num_of_words, word_vec_len)
    >>> trained_model[['office', 'products']]
    array([ -1.40128313e-02, ...]
          [ -1.70425311e-03, ...]
           ...)
    """
    def __init__(self):
        self.syn0 = []
        self.vocab = {}

    def word_vec(self, word):
        if word in self.vocab:
            return self.syn0[[self.vocab[word]]]
        else:
            raise KeyError("word '{0}' not in vocabulary".format(word))

    def __getitem__(self, words):
        if isinstance(words, str):
            return self.word_vec(words)

        return np.vstack([self.word_vec(word) for word in words])


def load_model(fname, encoding='utf8', unicode_errors='strict', datatype=np.float32):
    """Load pretrained Word2Vec file"""
    print("loading projection weights from {0}".format(fname))
    with smart_open(fname) as fin:
        header = str(fin.readline(), encoding=encoding)
        vocab_size, vector_size = map(int, header.split())  # throws for invalid file format

        result = Word2Vec()
        result.syn0 = np.zeros((vocab_size, vector_size), dtype=datatype)

        def add_word(word, weights):
            word_id = len(result.vocab)
            if word in result.vocab:
                print("duplicate word '%s' in %s, ignoring all but first".format(word, fname))
                return

            result.vocab[word] = word_id
            result.syn0[word_id] = weights


        binary_len = np.dtype(np.float32).itemsize * vector_size
        for line_no in range(vocab_size):
            # mixed text and binary: read text first, then binary
            word = []
            while True:
                ch = fin.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                if ch != b'\n':  # ignore newlines in front of words (some binary files have)
                    word.append(ch)
            word = str(b''.join(word), encoding=encoding, errors=unicode_errors)
            weights = np.fromstring(fin.read(binary_len), dtype=np.float32)
            add_word(word, weights)

    if result.syn0.shape[0] != len(result.vocab):
        print("duplicate words detected, shrinking matrix size from {0} to {1}".format(
                result.syn0.shape[0], len(result.vocab)
            ))
        result.syn0 = np.ascontiguousarray(result.syn0[: len(result.vocab)])
    assert (len(result.vocab), vector_size) == result.syn0.shape

    print("loaded {0} matrix from {1}".format(result.syn0.shape, fname))
    return result

