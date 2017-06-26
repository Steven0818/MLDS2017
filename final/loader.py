import numpy as np
import random
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.corpora.dictionary import Dictionary
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import util

class WikiAllData():
    def __init__(self, corpus, wiki_dict, wordfile, vocab_size=200000, window_size=5):
        self.w2id_dict = util.load_worddict(wordfile, vocab_size)
        self.window_size = window_size

        print('Starting loading Wiki Corpus...', end='')
        wiki_d = Dictionary.load(wiki_dict)
        self.wiki_corpus = WikiCorpus(corpus, dictionary=wiki_d)
        print('[done]')

    def batch_generator(self, batch_size=128):
        QUEUE_END = '__QUEUE_END105834569xx'

        def load(q, batch_size):
            text_gen = self.wiki_corpus.get_texts()
            input_batch = np.zeros(batch_size, dtype=np.int32)
            label_batch = np.zeros((batch_size, 1), dtype=np.int32)

            counter = 0 
            #w_vec : list of words in a document
            for w_vec in text_gen:
                id_vec = [self.w2id_dict[w] if w in self.w2id_dict else -1 for w in w_vec]
                init_mid = window = random.randint(2,self.window_size)
                
                # sliding window center go through the w_vec
                for mid_idx in range(init_mid, len(id_vec)-window-1):
                    start_idx = max(0,mid_idx-window)
                    end_idx = min(mid_idx+window, len(id_vec)-1)
                    # go through window
                    for target_idx in range(start_idx, end_idx+1):
                        if target_idx == mid_idx:
                            continue
                        if id_vec[target_idx] == -1:
                            continue

                        input_batch[counter] = id_vec[mid_idx]
                        label_batch[counter] = id_vec[target_idx]
                        
                        if counter == batch_size - 1:
                            q.put((input_batch, label_batch))
                            input_batch = np.zeros(batch_size, dtype=np.int32)
                            label_batch = np.zeros((batch_size, 1), dtype=np.int32)
                            counter = 0
                        else:
                            counter += 1
                    
                    window = random.randint(2, self.window_size)

            q.put(QUEUE_END)

        q = queue.Queue(maxsize=500)
        t = threading.Thread(target=load, args=(q, batch_size))
        t.daemon = True
        t.start()

        while True:
            q_output = q.get()
            if q_output == QUEUE_END:
                break
            yield q_output

class Wiki9Data():
    def __init__(self, corpus, w2id_dict, w2feq_dict, window_size=5):
        self.w2id_dict = w2id_dict
        self.wid2feq_dict = w2feq_dict
        self.window_size = window_size

        print('Starting loading Wiki Corpus...', end='')
        self.wiki_corpus = self._load_corpus(corpus)
        print('[done]')

    def _load_corpus(self, corpus):
        with open(corpus, 'r') as f:
            line = f.readlines()[0].split()
    
        id_vec = [self.w2id_dict[w] if w in self.w2id_dict else -1 for w in line]
        return id_vec

    def batch_generator(self, model_type='skipgram', batch_size=128):
        QUEUE_END = '__QUEUE_END105834569xx'

        def load(q, batch_size):
            input_batch = np.zeros(batch_size, dtype=np.int32)
            label_batch = np.zeros((batch_size, 1), dtype=np.int32)

            counter = 0 
            corpus_len = len(self.wiki_corpus)

            window = random.randint(2,self.window_size)

            # sliding window center go through the w_vec
            idx_queue = [i for i in range(corpus_len)]
            random.shuffle(idx_queue)
            for mid_idx in idx_queue:
                if self.wiki_corpus[mid_idx] == -1:
                    continue

                subsampling_prob = 1. - np.sqrt(0.00001 / self.wid2feq_dict[self.wiki_corpus[mid_idx]])
                if random.random() < subsampling_prob:
                    continue   

                start_idx = max(0,mid_idx-window)
                end_idx = min(mid_idx+window, corpus_len-1)

                # go through window
                for target_idx in range(start_idx, end_idx+1):
                    if target_idx == mid_idx:
                        continue
                    if self.wiki_corpus[target_idx] == -1:
                        continue
                    
                    if model_type == 'skipgram':
                        input_batch[counter] = self.wiki_corpus[mid_idx]
                        label_batch[counter] = self.wiki_corpus[target_idx]
                    elif model_type == 'cbow':
                        input_batch[counter] = self.wiki_corpus[target_idx]
                        label_batch[counter] = self.wiki_corpus[mid_idx]

                    if counter == batch_size - 1:
                        q.put((input_batch, label_batch))
                        input_batch = np.zeros(batch_size, dtype=np.int32)
                        label_batch = np.zeros((batch_size, 1), dtype=np.int32)
                        counter = 0
                    else:
                        counter += 1
                
                window = random.randint(2, self.window_size)

            q.put(QUEUE_END)

        q = queue.Queue(maxsize=500)
        t = threading.Thread(target=load, args=(q, batch_size))
        t.daemon = True
        t.start()

        while True:
            q_output = q.get()
            if q_output == QUEUE_END:
                break
            yield q_output

    def get_testdata(self):
        vec_A = np.array([self.w2id_dict[w] for w in ['king', 'men', 'son', 'great', 'possibly']], dtype=np.int32)
        vec_B = np.array([self.w2id_dict[w] for w in ['man', 'man', 'daughter', 'greater', 'impossibly']], dtype=np.int32)
        vec_C = np.array([self.w2id_dict[w] for w in ['woman', 'woman', 'granddaughter', 'tougher', 'unethical']], dtype=np.int32)
        vec_D = np.array([self.w2id_dict[w] for w in ['queen', 'car', 'grandson', 'tough', 'ethical']], dtype=np.int32)
        
        #vec_A = np.array([self.w2id_dict[w] for w in [b'king', b'men', b'son', b'great', b'possibly']], dtype=np.int32)
        #vec_B = np.array([self.w2id_dict[w] for w in [b'man', b'man', b'daughter', b'greater', b'impossibly']], dtype=np.int32)
        #vec_C = np.array([self.w2id_dict[w] for w in [b'woman', b'woman', b'granddaughter', b'tougher', b'unethical']], dtype=np.int32)
        #vec_D = np.array([self.w2id_dict[w] for w in [b'queen', b'car', b'grandson', b'tough', b'ethical']], dtype=np.int32)

        return vec_A, vec_B, vec_C, vec_D