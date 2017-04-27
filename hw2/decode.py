"""Module for decoding."""

import os
import time

import beam_search
import data
from six.moves import xrange
import tensorflow as tf
import util
import json
import eval
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_decode_steps', 1000000, 'Number of decoding steps.')


DECODE_LOOP_DELAY_SECS = 60


class DecodeIO(object):
    """Writes the decoded and references to RKV files for Rouge score.

        See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
    """

    def __init__(self, outdir):
        self._cnt = 0
        self._outdir = outdir
        if not os.path.exists(self._outdir):
            os.mkdir(self._outdir)
        self._decode_file = None

    def Write(self, decode):
        """Writes the reference and decoded outputs to RKV files.

        Args:
            decode: The machine-generated result
        """
        json.dump(decode, self._decode_file)
        self._cnt += 1

    def ResetFiles(self, global_step):
        """Resets the output files. Must be called once before Write()."""
        if self._decode_file:
            self._decode_file.close()

        self._decode_file = open(
                os.path.join(self._outdir, 'result{0}.json' % global_step), 'w')


class BSDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, test_data, beam_size, dict_rev):
        """Beam search decoding.

        Args:
            model: The seq2seq attentional model.
            batch_reader: The batch data reader.
            hps: Hyperparamters.
            vocab: Vocabulary
        """
        self._model = model
        self._model.build_graph()
        self.test_data = test_data
        self.beam_size = beam_size
        self.dict_rev = dict_rev
        self._saver = tf.train.Saver()
        self._decode_io = DecodeIO(FLAGS.decode_dir)

    def DecodeLoop(self):
        """Decoding loop for long running process."""
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        step = 0
        # while step < FLAGS.max_decode_steps:
        # 	time.sleep(DECODE_LOOP_DELAY_SECS)
        # 	if not self._Decode(self._saver, sess):
        # 		continue
        # 	step += 1
        self._Decode(self._saver, sess)

    def test_model(self):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
            return False

        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
                FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        self._saver.restore(sess, ckpt_path)
        x, video_ids, captions = self.test_data
        result = self._model.test_result(sess, x, captions)
        self.evaluate_output(None, captions, result)


    def _Decode(self, saver, sess):
        """Restore a checkpoint and decode it.

        Args:
            saver: Tensorflow checkpoint saver.
            sess: Tensorflow session.
        Returns:
            If success, returns true, otherwise, false.
        """
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to decode yet at %s', FLAGS.log_root)
            return False

        tf.logging.info('checkpoint path %s', ckpt_state.model_checkpoint_path)
        ckpt_path = os.path.join(
                FLAGS.log_root, os.path.basename(ckpt_state.model_checkpoint_path))
        tf.logging.info('renamed checkpoint path %s', ckpt_path)
        saver.restore(sess, ckpt_path)

        global_step = self._model.get_global_step(sess)
        self._decode_io.ResetFiles(global_step)

        results = []
        x, video_ids, captions = self.test_data
        bs = beam_search.BeamSearch(
            self._model, self.beam_size,
            util.BOS_ID,
            util.EOS_ID,
            45)
        for i in range(len(video_ids)):
            

            best_beam = bs.BeamSearch(sess, x[i])[0]
            decode_output = [int(t) for t in best_beam.tokens]
            results.append(decode_output)
            print(decode_output)

        score = self.evaluate_output(
                video_ids, captions, results)
        print('global_step {0} BLEU score: {1}'.format(global_step, score))
        return True

    def Ids2Words(self, ids):
        return [self.dict_rev[str(x)] for x in ids]
    

    def evaluate_output(self, video_ids, captions, results):
        """Convert id to words and writing results.
           Return avg. BLEU Score correspond to answers

        Args:
            video_ids: origin video id.
            captions: public test data ground truth with string type.
            result: The caption words ids output by machine.
        """
        score = 0
        answers = []
        for index, output_ids in enumerate(results):
            if util.EOS_ID in output_ids:
                output_ids = output_ids[:output_ids.index(util.EOS_ID)]
            decoded_output = ' '.join(self.Ids2Words(output_ids))
            score += np.mean(np.array([eval.BLEU(decoded_output, cap)
                                       for cap in captions[index]]))
            # answers.append({'id': video_ids[index], 'caption': decoded_output})
            
        # self._decode_io.Write(answers)
        return score / len(video_ids)
