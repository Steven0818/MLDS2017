import sys
import json
import time
import random
import numpy as np
import tensorflow as tf
import word2vec
from parallel_loader import ParallelLoader

N_THREAD = 1 if len(sys.argv) == 1 else int(sys.argv[1])

sentences = json.load(open('./data/data.json'))
subset = word2vec.load_indexed_model('./data/subset.bin')

SENTENCE_LEN = 5
BATCH_SIZE = 10

def loader_func():
    while True:
        ss = random.sample(sentences, BATCH_SIZE)
        ret = np.zeros((BATCH_SIZE, SENTENCE_LEN, 300))
        for i, x in enumerate(ss):
            a = subset[x]
            if a is None:
                continue
            len_x = min(SENTENCE_LEN, a.shape[0])
            ret[i][:len_x] = a[:len_x]
        yield ret

with tf.device('/cpu:0'):
    loader = ParallelLoader([None, SENTENCE_LEN, 300])
    inputs = loader.get_input(BATCH_SIZE)

s = tf.reduce_sum(inputs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf.train.start_queue_runners(sess=sess)
loader.start_loaders(sess, loader_func, n_loader=N_THREAD)

start_time = time.time()
for counter in range(10000):
    sess.run([s])

end_time = time.time()

print((end_time - start_time)/10000)
