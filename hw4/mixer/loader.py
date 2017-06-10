"""Data loader for Seq2Seq with buckets
Usage:
>>> data_dir = './data'
>>> buckets = [(5, 10), (10, 20), (20, 30), (30, 30)]
>>> data = Data(data_dir, buckets, convlen=2, batch_size=20)
Load 278468 lines
Load 3002 words
Load 63440 convs
>>> data.start_loaders(n=4)

# In each round, call data.get()
>>> a, x, mask = data.get()
>>> print(a, [m.shape for m in x])
0 [(5, 20), (10, 20)]
"""

import os
import random
from bisect import bisect_left
import multiprocessing as mp
import numpy as np
from worddict import WordDict

np.set_printoptions(precision=4, linewidth=150)

class Data:
    def __init__(self,
                 data_dir,
                 buckets,
                 convlen=2,
                 batch_size=20):
        self.convlen = convlen
        self.batch_size = batch_size
        self.queue = mp.Queue(maxsize=100)
        self.workers = []

        # Change internal buckets to line-ordered
        # So [(1, 2), (2, 3), (3, 4)] becomes [[1, 2, 3], [2, 3, 4]]
        self.buckets = np.array(buckets, dtype=np.int32).T

        # Word Dict {str: int} and {int: str}
        self.worddict = self.load_worddict(os.path.join(data_dir, 'worddict.txt'))
        print('Load {} words'.format(len(self.worddict)))

        # Line Dict {int: [int]}
        self.lines = self.load_linedict(os.path.join(data_dir, 'linedict.txt'))
        print('Load {} lines'.format(len(self.lines)))

        # Conversation List [[int]] or [[[int]]]
        self.convs, self.total_convs, self.total_convs_weight = self.load_convdict(
            os.path.join(data_dir, 'convdict.txt'))
        print('Load {} convs'.format(len(self.convs)))


    def reset(self, retain_convdict=False, retain_linedict=False):
        """
        Terminate all children process and flush queue
        """
        # convdict can be retained only if linedict is unchanged
        retain_convdict = retain_convdict and retain_linedict

        for p in self.workers:
            p.terminate()
        self.workers = []

        del self.queue
        self.queue = mp.Queue(maxsize=100)

        if not retain_convdict:
            del self.convs
            del self.total_convs
            del self.total_convs_weight

        if not retain_linedict:
            del self.lines


    def load_convdict(self, fpath):
        convdict = open(fpath).read().splitlines()
        convs = []
        cum_weight = [0]
        total_convs = [[] for _ in range(self.buckets.shape[1])]
        for conv in convdict:
            t = [int(x) for x in conv.split()]
            if len(t) < self.convlen:
                continue

            for x in (t[i:i+self.convlen] for i in range(len(t) - self.convlen)):
                bucket_id = self.get_bucket_id(x)
                total_convs[bucket_id].append(x)
            convs.append(t)

        for t in total_convs:
            cum_weight.append(len(t) + cum_weight[-1])

        conv = np.array(convs)
        return conv, total_convs, cum_weight


    def load_linedict(self, fpath):
        linedict = open(fpath).read().splitlines()
        lines = {}
        for line in linedict:
            t = [int(x) for x in line.split()]
            lines[t[0]] = t[2:]
        return lines


    def load_worddict(self, fpath):
        return WordDict.fromcsv(fpath)


    def get_bucket_id(self, convs):
        bucket_id = -1
        for i, line in enumerate(convs):
            new_bucket_id = bisect_left(self.buckets[i], len(self.lines[line]))
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
            random.randint(0, self.total_convs_weight[-1])) - 1
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
        mask = np.ones((len(lineids), bucket_len), dtype=np.int32)

        for i, line in enumerate(lineids):
            words = self.lines[line]
            ret[i, :len(words)] = words
            mask[i, len(words):] = 0

        ret = np.split(np.transpose(ret), 1)[0]
        mask = np.split(np.transpose(mask), 1)[0]

        return ret, mask

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
            masks = []

            for i in range(self.convlen):
                line, mask = self.get_line_batch(self.buckets[i, bucket_id],
                                                 convs[:, i])
                lines.append(line)
                masks.append(mask)

            self.queue.put((bucket_id, lines, masks))


    def __len__(self):
        if hasattr(self, 'total_convs_weight'):
            return self.total_convs_weight[-1]
        return 0
