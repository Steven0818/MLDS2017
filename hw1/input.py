import numpy as np
import multiprocessing as mp

class dataLoader():
    def __init__(self, input_words, num_steps=40, vocab_size=30000):
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        
        data_len = len(input_words)
        self.sentence_count = data_len // (num_steps + 1)
        
        self.sentence_arr = np.array(input_words[:sentence_count * (num_steps+1)]).reshape((sentence_count, num_steps+1))
        
    def batch_gen(self, batch_size):
        QUEUE_END = '__QUEUE_END105834569xx'
        
        def idx2vec(q):
            
            for i in range(self.sentence_count):
                vec = np.zeros([self.num_steps + 1, self.vocab_size])
                
                sentence = self.sentence_arr[i]
                for j, one_hot_idx in enumerate(sentence):
                    vec[j, one_hot_idx] = 1
                
                q.put(vec)
            
            q.put(QUEUE_END)
            q.close()
            
        q = mp.Queue(maxsize=256)
        
        p = mp.Process(target=idx2vec, args=(q))
        p.daemon = True
        p.start()
        
        x = np.zeros((batch_size, self.num_steps + 1, self.vocab_size))
        
        for i in xrange(0, sentence_count, batch_size):
            for j in xrange(batch_size):
                vec = q.get()
                
                if vec = QUEUE_END:
                    x = np.delete(x, range(j, batch_size), axis=0)
                    break
                
                x[j, ...] = vec
                
            yield x
       
        