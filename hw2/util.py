#!/usr/bin/python3

import numpy as np
import json
import re
from collections import Counter

replace_char =  r'[\"\'_:\+\-,()\[\]<>\\]'
VOCAB_COUNT = 3000


"""
Create dictionary mapping word string to word index.
The dictionary contains only top-{VOCAB_COUNT} most common word.    
"""

def build_word2idx_dict(vocab_size=3000,
                        trainlable_json='data/training_label.json', 
                        testlabel_json='data/testing_public_label.json',
                        dict_path='data/dict.json'):
        
    def _get_words_in_json(filepath):
        json_obj = json.load(open(filepath, 'r'))
        
        all_words = []
        for datum in json_obj:
            for sentence in datum['caption']:
                sentence = re.sub(replace_char, ' ', sentence).replace(".", " <eos>")
                words = [w.lower() for w in sentence.split()]
                
                if words[-1] != '<eos>':
                    words.append('<eos>')
                words.insert(0, '<bos>')
                
                all_words.extend(words)
    
        return all_words
    
    words = _get_words_in_json(trainlable_json) + _get_words_in_json(testlabel_json)
    
    word_counter = Counter(words)
    """
    There is 6085 words in this Counter.
    3772 words apppears at least 2 times.
    2936 words appears at least 3 times.
    """
    
    dictionary = {}
    for k, value in enumerate(word_counter.most_common(VOCAB_COUNT - 2)):
        dictionary[value[0]] = k+2
    # 0 for padding
    # 1 for unknown
        
    with open(dict_path, 'w') as f:
        json.dump(dictionary, f)


    
def get_tr_in_idx(trainlable_json='data/training_label.json', dict_path='data/dict.json'):
    d_word2idx = json.load(open(dict_path, 'r'))
    json_obj = json.load(open(trainlable_json, 'r'))
    
    new_json_obj = []
    
    for datum in json_obj:           
        new_datum = {} 
        new_datum['id'] = datum['id']
        new_datum['caption'] = []
        
        for sentence in datum['caption']:
            sentence = re.sub(replace_char, ' ', sentence).replace(".", " <eos>")
            words = [w.lower() for w in sentence.split()]
            
            if words[-1] != '<eos>':
                    words.append('<eos>')
            words.insert(0, '<bos>')
            
            new_datum['caption'].append([d_word2idx[w] if w in d_word2idx else 1 for w in words])
            
        new_json_obj.append(new_datum)
    
    return new_json_obj
    
    