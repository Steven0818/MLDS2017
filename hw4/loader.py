"""Data loader for Seq2Seq with buckets
Usage:
>>> data_dir = './data'
>>> buckets = [(5, 10), (10, 20), (20, 30), (30, 30)]
>>> data = Data(data_dir, buckets, convlen=2, batch_size=20)
>>> data.start_loaders(n=4)

# In each round, call data.get()
>>> a, x = data.get()
>>> print(a, [m.shape for m in x])
0 [(5, 20), (10, 20)]
"""

import os
import random
from bisect import bisect_left
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from worddict import WordDict

class Data:
    def __init__(self,
                 data_dir,
                 buckets,
                 convlen=2,
                 batch_size=20):
        self.convlen = convlen
        self.batch_size = batch_size

        # Change internal buckets to line-ordered
        # So [(1, 2), (2, 3), (3, 4)] becomes [[1, 2, 3], [2, 3, 4]]
        self.buckets = np.array(buckets, dtype=np.int32).T

        # Line Dict
        linedict = open(os.path.join(data_dir, 'linedict.txt')).read().splitlines()
        self.line = {}
        for line in tqdm(linedict, desc='Load lines'):
            t = [int(x) for x in line.split()]
            self.line[t[0]] = t[2:]

        # Word Dict
        self.worddict = WordDict.fromcsv(os.path.join(data_dir, 'worddict.txt'))
        print('Load {} words'.format(len(self.worddict)))

        # Conversation List
        convdict = open(os.path.join(data_dir, 'convdict.txt')).read().splitlines()
        convs = []
        total_convs = [[] for _ in buckets]
        for conv in tqdm(convdict, desc='Load conversations'):
            t = [int(x) for x in conv.split()]
            if len(t) < convlen:
                continue

            for x in (t[i:i+self.convlen] for i in range(len(t) - self.convlen)):
                bucket_id = self.get_bucket_id(x)
                total_convs[bucket_id].append(x)
            convs.append(t)

        self.conv = np.array(convs)
        self.total_convs = total_convs
        self.total_convs_weight = [0]
        for t in total_convs:
            self.total_convs_weight.append(len(t) + self.total_convs_weight[-1])

        self.queue = mp.Queue(maxsize=100)
        self.workers = []

    def get_bucket_id(self, convs):
        bucket_id = -1
        for i, line in enumerate(convs):
            new_bucket_id = bisect_left(self.buckets[i], len(self.line[line]))
            bucket_id = max(new_bucket_id, bucket_id)
        return bucket_id

    def start_loaders(self, n=4):
        for _ in range(n):
            p = mp.Process(target=self.add_to_queue)
            p.start()
            self.workers.append(p)

    def get(self):
        return self.queue.get() # NOTICE: it is blocking

    def get_conv_batch(self, batch_size):
        """
        Return a numpy array with shape (batch_size, convlen). Each row in the
        array is a randomly sampled conversation (its starting line-id is also
        randomly choosed).
        """
        bucket_id = bisect_left(
            self.total_convs_weight,
            np.random.randint(0, self.total_convs_weight[-1])) - 1
        ret = random.choices(self.total_convs[bucket_id], k=batch_size)

        return bucket_id, np.array(ret)

    def get_line_batch(self, bucket_len, lineids):
        """
        Input:
            i: the i-th line in the conversation
            bucket_len: the length to be padded to
            lineids: a list of line-ids with length L
        Return: np array with shape (bucket_len, L)
        """
        # Create the 2d batch-major array and transpose it in the end
        ret = np.zeros((len(lineids), bucket_len), dtype=np.int32)

        for i, line in enumerate(lineids):
            words = self.line[line]
            ret[i, :len(words)] = words
        return ret.T

    def add_to_queue(self):
        """
        This function is called by worker processes.

        In each round, it add a tuple
        (bucket_id, np arrays) into queue, where:
            bucket_id: can be used in Seq2Seq model
            np arrays: each np array has shape (bucket_len, batch_size)
        """
        while True:
            bucket_id, convs = self.get_conv_batch(self.batch_size)
            lines = []

            for i in range(self.convlen):
                line = self.get_line_batch(self.buckets[i, bucket_id],
                                           convs[:, i])
                lines.append(line)

            self.queue.put((bucket_id, lines))
