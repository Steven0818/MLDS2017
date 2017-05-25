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
    def __init__(self, img_dir, img_shape, caption_json, n_eye_types=11, n_hair_types=11):
        self.file_ids, self.filepaths = self._get_img_path(img_dir)
        self.n_data = len(self.filepaths)
        self.img_shape = img_shape
        self.id2cap_dict = self._load_caption(caption_json)
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

        eyes_onehot = np.zeros(self.n_eye_types, dtype=np.float32)
        hair_onehot = np.zeros(self.n_hair_types, dtype=np.float32)

        wrong_eyes_onehot = np.zeros(self.n_eye_types, dtype=np.float32)
        wrong_hair_onehot = np.zeros(self.n_hair_types, dtype=np.float32)

        eyes_idx_list = self.id2cap_dict[ids]['eyes']
        for idx in eyes_idx_list:
            eyes_onehot[idx] = 1.
            
            wrong_idx = idx
            while True:
                wrong_idx = wrong_idx + 1 if wrong_idx + 1 < self.n_eye_types else 0
                if not wrong_idx in eyes_idx_list:
                    break
            wrong_eyes_onehot[wrong_idx] = 1.
        
        hair_idx_list = self.id2cap_dict[ids]['hair']
        for idx in hair_idx_list:
            hair_onehot[idx] = 1.
            
            wrong_idx = idx
            while True:
                wrong_idx = wrong_idx + 1 if wrong_idx + 1 < self.n_hair_types else 0
                if (not wrong_idx in hair_idx_list) or wrong_idx == idx:
                    break
            wrong_hair_onehot[wrong_idx] = 1.

        correct_vec = np.concatenate((eyes_onehot, hair_onehot))
        wrong_vec = np.concatenate((wrong_eyes_onehot, wrong_hair_onehot))

        return (norm_im, correct_vec, wrong_vec)
    
class DataLoader2:
    def __init__(self, img_dir, img_shape, embedding_dim, fidx2arrIdx_json, tags_npy):
        self.file_ids, self.filepaths = self._get_img_path(img_dir)
        self.n_data = len(self.filepaths)
        self.img_shape = img_shape
        self.fidx2arridx_dict = self._load_caption(fidx2arrIdx_json)
        self.embedding_arr = np.load(tags_npy)
        self.embedding_dim = embedding_dim

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

                    images, correct_tags = zip(*list(pool.map(self._load_image, batch_paths, batch_ids)))
                    imgs = np.array(images)
                    correct_tags = np.array(correct_tags)
                    wrong_tags = np.zeros(correct_tags.shape)
                    wrong_tags[0:-1] = correct_tags[1:]
                    wrong_tags[-1] = correct_tags[0]
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

        tag_idx = self.fidx2arridx_dict[ids]
        correct_vec = self.embedding_arr[tag_idx]

        return (norm_im, correct_vec)