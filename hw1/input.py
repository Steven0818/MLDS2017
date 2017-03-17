import numpy as np
import multiprocessing as mp
import random


class DataLoader():
    def __init__(self, input_words, num_steps=40, vocab_size=30000):
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        
        data_len = len(input_words)
        self.sentence_count = data_len // (num_steps)
        
        self.sentence_arr = np.array(input_words[:self.sentence_count * (num_steps)]).reshape((self.sentence_count, num_steps))
        
        
    def shuffle(self):
        np.random.shuffle(self.sentence_arr)
    
    def batch_count(self, batch_size):
        return (self.sentence_arr.shape[0] // batch_size) + 1
    
    def batch_gen(self, batch_size):
        QUEUE_END = '__QUEUE_END105834569xx'
        
        def idx2vec(num_steps, vocab_size, q):
            
            for i in range(self.sentence_count):
                vec = np.zeros([num_steps, vocab_size])
                
                sentence = self.sentence_arr[i]
                for j, one_hot_idx in enumerate(sentence):
                    vec[j, one_hot_idx] = 1
                
                q.put(vec)
            
            q.put(QUEUE_END)
            q.close()
            
        q = mp.Queue(maxsize=100)
        
        p = mp.Process(target=idx2vec, args=(self.num_steps, self.vocab_size, q))
        p.daemon = True
        p.start()
        
        x = np.zeros((batch_size, self.num_steps, self.vocab_size))
        
        for i in range(0, self.sentence_count, batch_size):
            for j in range(batch_size):
                vec = q.get()
                
                if vec == QUEUE_END:
                    x = np.delete(x, range(j, batch_size), axis=0)
                    break
                
                x[j, ...] = vec
                
            yield x
       
class DataLoader2():
    def __init__(self, input_words, num_steps=40, vocab_size=30000):
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        
        data_len = len(input_words)
        self.sentence_count = data_len // (num_steps)
        
        self.sentence_arr = np.array(input_words[:self.sentence_count * (num_steps)]).reshape((self.sentence_count, num_steps))
        
        
    def shuffle(self):
        np.random.shuffle(self.sentence_arr)
    
    def batch_count(self, batch_size):
        return (self.sentence_arr.shape[0] // batch_size) + 1
    
    def batch_gen(self, batch_size):
        QUEUE_END = '__QUEUE_END105834569xx'
        
        def idx2vec(num_steps, vocab_size, batch_size, q):            
            batch_count_epoch = self.sentence_count // batch_size + 1
      
            for i in range(batch_count_epoch):
                x = np.zeros([batch_size, num_steps, vocab_size])
                for j in range(batch_size):
                    sentence_idx = i*batch_size + j
                    
                    if sentence_idx < self.sentence_count:
                        sentence = self.sentence_arr[sentence_idx]
                        for k, one_hot_idx in enumerate(sentence):
                            x[j, k, one_hot_idx] = 1
                    else: 
                        x = np.delete(x, range(j, batch_size), axis=0)
                        break
                        
                q.put(x)
                    
            q.put(QUEUE_END)
            q.close()
            
        q = mp.Queue(maxsize=20)
        
        p = mp.Process(target=idx2vec, args=(self.num_steps, self.vocab_size, batch_size, q))
        p.daemon = True
        p.start()
        
        x = np.zeros((batch_size, self.num_steps, self.vocab_size))
        
        
        batch_count_epoch = self.sentence_count // batch_size + 1
        for i in range(batch_count_epoch):
            x = q.get()
                
            if x == QUEUE_END:  
                break
                
            yield x
        
