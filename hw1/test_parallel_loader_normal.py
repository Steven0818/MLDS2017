import json
import time
import random
import numpy as np
import tensorflow as tf
import word2vec
from parallel_loader import ParallelLoader

sentences = json.load(open('./data/data.json'))
subset = word2vec.load_indexed_model('./data/subset.bin')

SENTENCE_LEN = 5
BATCH_SIZE = 10

def loader_func():
    ss = random.sample(sentences, BATCH_SIZE)
    ret = np.zeros((BATCH_SIZE, SENTENCE_LEN, 300))
    for i, x in enumerate(ss):
        a = subset[x]
        if a is None:
            continue
        len_x = min(SENTENCE_LEN, a.shape[0])
        ret[i][:len_x] = a[:len_x]
    return ret
"""
with tf.device('/cpu:0'):
    loader = ParallelLoader([None, SENTENCE_LEN, 300])
    inputs = loader.get_input(BATCH_SIZE)
"""
inputs = tf.placeholder(shape=[None, SENTENCE_LEN, 300], dtype=np.float32)
s = tf.reduce_sum(inputs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
"""
tf.train.start_queue_runners(sess=sess)
loader.start_loaders(sess, loader_func)
"""
start_time = time.time()
for counter in range(100000):
    sess.run([s], feed_dict={inputs: loader_func()})

end_time = time.time()

print((end_time - start_time)/100000)
