#!/usr/bin/python3
import os
import numpy as np
import json
import re
import random
from collections import Counter
from tqdm import tqdm
import logging
import multiprocessing as mp

replace_char =  r'[\"\'_:\+\-,()\[\]<>\\]'
VOCAB_COUNT = 3000

PAD_ID = 0
UNK_ID = 1
BOS_ID = 4


"""
Create dictionary mapping word string to word index.
The dictionary contains only top-{VOCAB_COUNT} most common word.    
"""

def build_word2idx_dict(vocab_size=3000,
                        trainlable_json='data/training_label.json', 
                        testlabel_json='data/testing_public_label.json',
                        dict_path='data/dict.json',
                        dict_rev_path='data/dict_rev.json'):
        
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
    
    dictionary = dict()
    dictionary_rev = dict()
    for k, value in enumerate(word_counter.most_common(VOCAB_COUNT - 2)):
        dictionary[value[0]] = k+2
        dictionary_rev[int(k+2)] = value[0]
    # 0 for padding
    # 1 for unknown
        
    with open(dict_path, 'w') as f:
        json.dump(dictionary, f)
    with open(dict_rev_path, 'w') as f:
        json.dump(dictionary_rev, f)


    
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


class Data:
    def __init__(self, feat_dir, training_json, word2idx, idx2word, batch_size, shuffle=True):
        self.feat_dir = feat_dir
        self.training_json = training_json
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.data = []
        self.workers = []
        self.batch_size = batch_size

        for t in tqdm(training_json):
            _id, _captions = self.get_one_caption_list(t)

            _feat = self.get_feat(_id)
            if _feat is None:
                continue

            self.data.extend([(_feat, c) for c in _captions])

        if shuffle:
            random.shuffle(self.data)

    def get_one_caption_list(self, datum):
        _id = datum['id']
        _captions = []

        for sentence in datum['caption']:
            sentence = re.sub(replace_char, ' ',
                              sentence).replace(".", " <eos>")
            words = [w.lower() for w in sentence.split()]

            if words[-1] != '<eos>':
                words.append('<eos>')

            _captions.append(
                [self.word2idx[w] if w in self.word2idx else UNK_ID for w in words])
            _captions[-1].insert(0, BOS_ID)
        return _id, _captions

    def get_feat(self, cid):
        _cid = '{0}.npy'.format(cid)
        if _cid in os.listdir(self.feat_dir):
            return np.load(os.path.join(self.feat_dir, _cid))

        return None

    def loader(self):
        def put_batch(data, b, q):
            while True:
                candidates = random.sample(data, k=b)
                q.put(Data.get_batch(candidates))

        q = mp.Queue(maxsize=200)

        for i in range(5):
            p = mp.Process(name='worker{0}'.format(i),
                           target=put_batch,
                           args=(self.data, self.batch_size, q))
            p.daemon = True
            p.start()
            self.workers.append(p)

        while True:
            yield q.get()

    @staticmethod
    def get_batch(data):
        batch_size = len(data)
        frames = [np.zeros((20, 4096)) for _ in range(batch_size)]
        captions = [[0] * 46 for _ in range(batch_size)]
        target_weights = [[0.0] * 46 for _ in range(batch_size)]

        for i, (frame, caption) in enumerate(data):
            for j in range(20):
                frames[i][j] = frame[j*4]
            for j, word in enumerate(caption):
                captions[i][j] = word
                target_weights[i][j] = 1.0

        return frames, captions, target_weights


    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        return self.data[x]
