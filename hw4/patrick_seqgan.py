# pylint: disable=E1101
import os
import sys
import random

import numpy as np
import tensorflow as tf
from tqdm import trange
from fire import Fire

from mixer.loader import Data
from mixer.utils import Config, AverageMeter

from mixer.seq2seq_model import Seq2SeqModel

PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(PROJECT_DIR)

class Main:
    def __init__(self,
                 config,
                ):
        config = Config(config)
        self.config = config
        self.data = Data(config.datadir,
                         config.generator['buckets'],
                         convlen=2,
                         batch_size=config.generator['batch_size'])

        self.worddict_len = len(self.data.worddict)
        self.generator = Seq2SeqModel(self.worddict_len,
                                      self.worddict_len,
                                      config.generator['buckets'],
                                      config.generator['hidden_dim'],
                                      1,
                                      10,
                                      config.generator['batch_size'],
                                      config.generator['lr'],
                                      config.generator['lr_decay'],
                                     )
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def pretrain_generator(self, epoch):
        self.sess.run(tf.global_variables_initializer())
        self.data.start_loaders(n=4)
        loss_meter = AverageMeter()
        config = self.config
        runs = int(len(self.data) / config.generator['batch_size'])
        for i in range(epoch):
            for _ in trange(runs):
                bucket_id, [req, res], [_, res_mask] = self.data.get()

                loss = self.generator.pretrain(self.sess,
                                               req,
                                               res,
                                               res_mask,
                                               bucket_id)
                loss_meter.update(loss)

            if i % config.generator['print_freq']:
                print('Epoch: {0} Training Loss: {1}'.format(i, loss_meter.avg))
                for result in self.step()[:5]:
                    print(' '.join(self.data.worddict.batch_get_inv(result)))

        self.saver.save(self.sess, config.generator['pretrain'])

    def pretrain_discriminator(self):
        pass

    def step(self):
        self.sess.run(tf.global_variables_initializer())
        self.data.start_loaders(n=4)
        bucket_id, [req, res], _ = self.data.get()

        _, results = self.generator.step(self.sess,
                                         req,
                                         res,
                                         bucket_id)
        return results

    def train(self):
        pass

if __name__ == '__main__':
    Fire(Main)
