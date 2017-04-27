import numpy as np
import multiprocessing as mp
import random
import os

class DataLoader():
    def __init__(self, input_json, data_path='data/training_data/feat' ,frame_step=20, frame_dim=4096, caption_step=45 ,vocab_size=3000):
        self.vocab_size = vocab_size
        self.data_path = data_path
        self.frame_step = frame_step
        self.frame_dim = frame_dim
        self.caption_step = caption_step
        
        self.video_names = []
        self.cap_sentences = []
        for video in input_json:
            for sentence in video['caption']:
                self.video_names.append(video['id'])
                self.cap_sentences.append(sentence)
        self.shuffle()

    def shuffle(self):
        z = list(zip(self.video_names, self.cap_sentences))
        random.shuffle(z)
        self.video_names, self.cap_sentences = zip(*z)
        
    def batch_gen(self, batch_size):
        QUEUE_END = '__QUEUE_END105834569xx'
        
        def raw2vec(caption_step, q):
            
            for i, filename in enumerate(self.video_names):
                filepath = os.path.join(self.data_path, filename + '.npy')
                x = np.load(filepath)                
                y = np.array(self.cap_sentences[i]).astype(np.int32)
                q.put((x,y))
            q.put(QUEUE_END)
            q.close()
            
        q = mp.Queue(maxsize=200)
        
        p = mp.Process(target=raw2vec, args=(self.caption_step, q))
        p.daemon = True
        p.start()
        
        x_batch = np.zeros((batch_size, self.frame_step, self.frame_dim), dtype=np.float32)
        y_batch = np.zeros((batch_size, self.caption_step+1), dtype=np.int32)
        y_mask = np.zeros((batch_size, self.caption_step+1), dtype=np.int32)
        
        for i in range(0, len(self.video_names), batch_size):
            for j in range(batch_size):
                vec = q.get()
                
                if vec == QUEUE_END:
                    x_batch = np.delete(x_batch, range(j, batch_size), axis=0)
                    y_batch = np.delete(y_batch, range(j, batch_size), axis=0)
                    y_mask = np.delete(y_mask, range(j, batch_size), axis=0)
                    break
                x, y = vec
                x = np.asarray([x[k*2,:]for k in range(self.frame_step)])
                x_batch[j, ...] = x
                y_batch[j,:len(y)] = y
                y_mask[j, :len(y)] = 1
                
            yield x_batch, y_batch,y_mask

class TestDataLoader():
    def __init__(self, input_json, data_path='data/testing_data/feat', frame_step=20, frame_dim=4096, caption_step=45, vocab_size=3000, shuffle=True):
        self.vocab_size = vocab_size
        self.data_path = data_path
        self.frame_step = frame_step
        self.frame_dim = frame_dim
        self.caption_step = caption_step

        self.video_names = []
        self.captions = []
        for video in input_json:
                self.video_names.append(video['id'])
                self.captions.append([sen.replace('.', '') for sen in video['caption']])
    def get_data(self, batch_size):
        ret = []


        for i in range(0, len(self.video_names), batch_size):
            end = i + batch_size
            x_batch = np.zeros((batch_size, self.frame_step, self.frame_dim), dtype=np.float32)
            for j in range(batch_size):     
                if i + j >= len(self.video_names):
                    x_batch = np.delete(x_batch, range(j, batch_size), axis=0)
                    end = i + j
                    break
                filename = self.video_names[i + j]
                filepath = os.path.join(self.data_path, filename + '.npy')
                x = np.load(filepath) 
                x = np.asarray([x[k*2,:]for k in range(self.frame_step)])
                x_batch[j, ...] = x
            ret.append((x_batch, self.video_names[i:end], self.captions[i:end]))
        return ret

    def get_all_data(self):

        x_batch = np.zeros((len(self.video_names), self.frame_step,
                            self.frame_dim), dtype=np.float32)
        for i in range(len(self.video_names)):

            filename = self.video_names[i]
            filepath = os.path.join(self.data_path, filename + '.npy')
            x = np.load(filepath)
            x = np.asarray([x[k * 2, :]for k in range(self.frame_step)])
            x_batch[i, ...] = x

        return (x_batch, self.video_names, self.captions)
