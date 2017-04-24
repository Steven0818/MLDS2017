import sys
import json
import numpy as np
import model
import bs_model

import util
import input
import eval
import time
from beam_search import BeamSearch
import tensorflow as tf

VOCAB_SIZE = 3000
FRAME_STEP = 20
FRAME_DIM = 4096
BATCH_SIZE = 100
CAPTION_STEP = 45
EPOCH = 1000
SCHEDULED_SAMPLING_CONVERGE = 5000
BEAM_SIZE = 4

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_root', './model', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', './train', 'Directory for train.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('beam_size', 4,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 300, 'How often to checkpoint.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')


train_npy_path = 'data/training_data/feat'


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
    """Calculate the running average of losses."""
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)
    loss_sum = tf.Summary()
    loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    sys.stdout.write('running_avg_loss: %f\n' % running_avg_loss)
    return running_avg_loss

def Train(model, data_batcher):
    """Runs model training."""
    model.build_graph()
    saver = tf.train.Saver()
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir)
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                            is_chief=True,
                            saver=saver,
                            summary_op=None,
                            save_summaries_secs=60,
                            save_model_secs=FLAGS.checkpoint_secs,
                            global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))
    running_avg_loss = 0
    step = 0
    while not sv.should_stop() and step < FLAGS.max_run_steps:
        frames, captions, loss_weights = next(data_batcher)
        (_, summaries, loss, train_step) = model.run_train_step(
            sess, frames, captions, loss_weights)

        summary_writer.add_summary(summaries, train_step)
        running_avg_loss = _RunningAvgLoss(
            running_avg_loss, loss, summary_writer, train_step)
    step += 1
    if step % 100 == 0:
        summary_writer.flush()
    sv.Stop()
    return running_avg_loss


def main():


    print ("building model...")
    # S2VT = model.Effective_attention_model(caption_steps=CAPTION_STEP)
    # S2VT.initialize()
    print ("building model successfully...")
    # bm = BeamSearch(S2VT, BEAM_SIZE, util.BOS_ID, util.EOS_ID, CAPTION_STEP)
    
    d_word2idx = json.load(open('data/dict.json', 'r'))
    d_idx2word = json.load(open('data/dict_rev.json', 'r'))
    tr_in_idx = util.get_tr_in_idx(trainlable_json='data/training_label.json', dict_path='data/dict.json')
    test_label = json.load(open('data/testing_public_label.json'))

    data = util.Data(
        'data/training_data/feat',
        json.load(open('data/training_label.json')),
        d_word2idx,
        d_idx2word,
        BATCH_SIZE)
    test_data_loader = input.TestDataLoader(test_label,
                                        data_path='data/testing_data/feat',
                                        frame_step=FRAME_STEP,
                                        frame_dim=FRAME_DIM,
                                        caption_step=CAPTION_STEP,
                                        vocab_size=VOCAB_SIZE,
                                        shuffle=False
                                        )
    train_label = json.load(open('data/training_label.json'))
    train_test_data_loader = input.TestDataLoader(train_label,
                                                  data_path = 'data/training_data/feat',
                                                  frame_step = FRAME_STEP,
                                                  frame_dim = FRAME_DIM,
                                                  caption_step=CAPTION_STEP,
                                                  vocab_size=VOCAB_SIZE,
                                                  shuffle=False)
    test_batch = test_data_loader.get_data(BATCH_SIZE)
    train_test_batch = train_test_data_loader.get_data(BATCH_SIZE)

    global_step = 0
    epoch_count = 1
    epoch_size = len(data)
    loader = data.loader()
    print ("training start....")
    model = bs_model.Beamsearch_attention_model()
    Train(model, loader)
    """
    for i in range(int(len(data) * EPOCH / BATCH_SIZE)):  
        
        frames, captions, target_weights = next(loader)
        start_time = time.time()
        cost = S2VT.train(
            frames, captions, target_weights, scheduled_sampling_prob=i / SCHEDULED_SAMPLING_CONVERGE)
        global_step += 1
        finish_time = time.time()
        #print ('each step time cost: {0}'.format(finish_time-start_time))
        test_bm(bm, test_batch, d_idx2word, global_step)
        if global_step % 100 == 0:
            print('global_step {0} cost: {1}'.format(global_step, cost))
        if global_step % 1000 == 0:
            util.test(S2VT, test_batch, d_idx2word, global_step,train_test = 'test')
            util.test(S2VT, train_test_batch, d_idx2word, global_step,train_test = 'train')
        if i * BATCH_SIZE > epoch_count * epoch_size:
            print('Epoch {0} end'.format(epoch_count))
            epoch_count += 1
    """


if __name__ == '__main__':
    main()
