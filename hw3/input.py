import numpy as np
import random
from PIL import Image
import os
from os import path
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

class DataLoader:
    def __init__(self, img_dir, img_shape):
        self.file_ids, self.filepaths = self._get_img_path(img_dir)
        self.n_data = len(self.filepaths)
        self.img_shape = img_shape
        
    def _get_img_path(self, img_dir):
        paths = []
        ids = []
        for name in os.listdir(img_dir):
            paths.append(path.join(img_dir, name))
            ids.append(name.split('.')[0])
        return ids, paths

    def shuffle(self):
        z = list(zip(self.filepaths, self.file_ids))
        random.shuffle(z)
        self.filepaths, self.file_ids = zip(*z)

    def batch_generator(self, batch_size=100):
        QUEUE_END = '__QUEUE_END105834569xx'
        
        def load(q, batch_size):     
            with ThreadPoolExecutor(max_workers=25) as pool:
                for i in range(0, self.n_data, batch_size): 
                    if i + batch_size <= self.n_data:
                        end_idx = i + batch_size
                        size = batch_size
                    else:
                        end_idx = self.n_data
                        size = self.n_data - i
                        q.put(QUEUE_END)
                        break
                        
                    batch_paths = self.filepaths[i:end_idx]

                    images = list(pool.map(self._load_image, batch_paths))
                    imgs = np.array(images)
                    q.put(imgs)
            
        q = queue.Queue(maxsize=30)
        t = threading.Thread(target=load, args=(q, batch_size))
        t.daemon = True
        t.start()
            
        for i in range(0, self.n_data, batch_size):
            imgs = q.get()
            if imgs == QUEUE_END:
                break
                
            yield imgs
    
    def _load_image(self, imgpath):
        im = Image.open(imgpath)
        return (np.asarray(im, dtype=np.float32) - 127.5) / 127.5
    
