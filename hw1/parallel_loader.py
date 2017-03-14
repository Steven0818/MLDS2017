"""
This module implement parallel loading to accelerate training and
reduce memory usage at the same time.

This code is modified from
https://indico.io/blog/tensorflow-data-input-part2-extensions/

IMPORTANT NOTES

Constructing Tensorflow graph is NOT thread-safe. Therefore, the
correct initialization order is

1. Construct the training graph
2. Call tf.initialize_all_variables()
3. Start our loader threads
4. Do training

# Doing anything with data on the CPU is generally a good idea.

>> with tf.device("/cpu:0"):
>>     data_loader = ParallelLoader()
>>     seq = data_loader.get_inputs()

See `test_parallel_loader.py` for toy example

"""

import threading
import numpy as np
import tensorflow as tf

class ParallelLoader:
    def __init__(self, input_shape, dtype=np.float32, capacity=500, min_after_dequeue=100):
        self.data = tf.placeholder(dtype=dtype, shape=input_shape)
        self.queue = tf.RandomShuffleQueue(shapes=[input_shape if input_shape[0] != None else input_shape[1:]],
                                           dtypes=[dtype],
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)

        self.enqueue_op = self.queue.enqueue_many([self.data])

    def loader_main(self, sess, loader_func):
        for data in loader_func():
            sess.run(self.enqueue_op, feed_dict={self.data: data})

    def get_input(self, batch_size):
        return self.queue.dequeue_many(batch_size)

    def start_loaders(self, sess, loader_func, n_loader=1):
        threads = []
        for n in range(n_loader):
            t = threading.Thread(target=self.loader_main, args=(sess, loader_func))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
