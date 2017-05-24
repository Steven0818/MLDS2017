import numpy as np
import random
import cv2
import os
from os import path
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

class DataLoader:
    def __init__(self, img_dir, img_shape, feature_path, w_feature_path, index_order_path, n_eye_types=11, n_hair_types=11):
        self.file_ids, self.filepaths = self._get_img_path(img_dir)
        self.n_data = len(self.filepaths)
        self.img_shape = img_shape
        self.load_embedding_feature(feature_path, w_feature_path, index_order_path)
        self.n_eye_types = n_eye_types
        self.n_hair_types = n_hair_types

    def _get_img_path(self, img_dir):
        paths = []
        ids = []
        for name in os.listdir(img_dir):
            paths.append(path.join(img_dir, name))
            ids.append(name.split('.')[0])
        return ids, paths

    def _load_caption(self, filepath):
        with open(filepath, 'r') as f:
            tags_json = json.load(f)
        
        return tags_json

    def load_embedding_feature(self, feature, w_feature, index_order):
        self.order = json.load(open(index_order))
        self.feature = np.load(feature)
        self.w_feature = np.load(w_feature)

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
                        q.put((QUEUE_END, QUEUE_END, QUEUE_END))
                        break
                        
                    batch_paths = self.filepaths[i:end_idx]
                    batch_ids = self.file_ids[i:end_idx]

                    images, correct_tags, wrong_tags = zip(*list(pool.map(self._load_image, batch_paths, batch_ids)))
                    imgs = np.array(images)
                    correct_tags = np.array(correct_tags)
                    wrong_tags = np.array(wrong_tags)
                    q.put((imgs, correct_tags, wrong_tags))
            
        q = queue.Queue(maxsize=30)
        t = threading.Thread(target=load, args=(q, batch_size))
        t.daemon = True
        t.start()
            
        for i in range(0, self.n_data, batch_size):
            imgs, correct_tags, wrong_tags = q.get()
            if imgs == QUEUE_END:
                break
                
            yield imgs, correct_tags, wrong_tags
    
    def _load_image(self, imgpath, ids):
        im = cv2.imread(imgpath)
        resize_im = cv2.resize(im, (64,64), interpolation=cv2.INTER_CUBIC)
        norm_im = (resize_im.astype(np.float32) - 127.5) / 127.5
        order = self.order[str(ids)]
        return (norm_im, self.feature[order], self.w_feature[order])
    
