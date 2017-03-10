from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from collections import Counter
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer

word_tokenizer = TweetTokenizer()
# word_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = SnowballStemmer("english")

class DataSet(object):
    """
    all prepocessing method and get data
    """



def build_dict(data_path="./Holmes_Training_Data", dict_path="./dict.json"):
    ## word count
    word_count = Counter()
    for filename in os.listdir(data_path):
        read_count = 0
        with open(os.path.join(data_path,filename), encoding="utf-8", errors='ignore') as f:
            words = []
            for line in f:
                # trim LICENSE
                if line.find("*END*THE SMALL PRINT!") != -1:
                    continue
                tokens = ([stemmer.stem(plural) for plural
                           in word_tokenizer.tokenize(line)])
                words.extend(tokens)
            word_count.update(Counter(words))
            read_count += 1
            print(read_count, "file read")
    ## build dict
    dictionary = {}
    ## 0 is others
    for k, value in enumerate(word_count.most_common()):
        dictionary[value[0]] = k+1
    with open(dict_path, 'w') as f:
        json.dump(dictionary, f)

def load_dictionary(size, dict_path="./dict.json"):
    if os.path.exists(dict_path):
        print("dictionary exists load...")
    else:
        print("dictionary does not exist build...")
        build_dict()
    with open(dict_path, "r") as f:
        json_dict = json.load(f)
    return {k: v for k, v in json_dict.items() if v < size}



    # def chunks(l, n):
    #     """Yield successive n-sized chunks from l."""
    #     for i in range(0, len(l), n):
    #         yield l[i:i + n]

def one_of_n_encode(N=30000, data_path="./Holmes_Training_Data"):
    dictionary = load_dictionary(N)
    def to_index(token):
        if token in dictionary:
            return dictionary[token]
        else:
            return 0
    def sentence_to_index_arr(sen):
        return [to_index(stemmer.stem(x)) for x in word_tokenizer.tokenize(sen)]

    sentences = []
    for filename in os.listdir(data_path):
        with open(os.path.join(data_path, filename), encoding="utf-8", errors='ignore') as f:
            # trim LICENSE
            for line in f:
                if line.find("*END*THE SMALL PRINT!") != -1:
                    break
            data = f.read().replace('\n', ' ')
            sentences.extend([sentence_to_index_arr(t) for t in sent_tokenize(data)])
    return sentences
