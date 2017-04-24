from collections import namedtuple
import numpy as np
from six.moves import xrange
import tensorflow as tf
import seq2seq_lib

HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps, '
                     'min_input_len, num_hidden, emb_dim, max_grad_norm, '
                     'num_softmax_samples')

def extract_argmax_and_embed_with_scheduled(embedding, ground_truth, output_projection=None, scheduled_prob=1.0, update_embedding=False):
    """Get a loop_function that extracts the previous symbol and embeds it with a scheduled sampling prob.
    Args:
        embedding: embedding tensor for symbols.
        ground_truth: ground_truth for scheduled sampling
        output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.
    Returns:
        A loop function.
    """
    def loop_function(prev, i):
        """function that feed previous model output rather than ground truth with a scheduled_sampling_prob."""
        prev_symbol = tf.cond(scheduled_prob >= tf.random_uniform([], 0, 1, dtype=tf.float64),
                            lambda: ground_truth[min(i+1, len(ground_truth)-1)],
                            lambda: tf.argmax(tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1]), 1))
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev
    return loop_function

class Beamsearch_attention_model():
    def __init__(self,
                 frame_steps=20,
                 frame_feat_dim=4096,
                 caption_steps=45,
                 vocab_size=3000,
                 dim_hidden=200,
                 schedule_sampling_converge=500,
                 batch_size=100,
                 enc_layers=4,
                 num_softmax_samples=1500,
                 mode='train'):
        self.frame_steps = frame_steps
        self.frame_feat_dim = frame_feat_dim
        self.caption_steps = caption_steps
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.enc_layers = enc_layers
        self.mode = mode
        self.num_softmax_samples = num_softmax_samples
        self.lr = 0.15
        self.min_lr = 0.01
        self.max_grad_norm = 2
        self.scheduled_sampling_decay = 10000

        self.global_step = tf.Variable(0, trainable=False)

    def add_placeholders(self):
        self.frames = tf.placeholder(
            tf.float32, [self.batch_size, self.frame_steps, self.frame_feat_dim], name='frame_input')
        self.captions = tf.placeholder(
            tf.int64, [self.batch_size, self.caption_steps+1])
        self.loss_weights = tf.placeholder(
            tf.float32, [self.batch_size, self.caption_steps+1])

        self.keep_prob = tf.placeholder(tf.float32, name='dropout_prob')

    def add_seq2seq(self):
        with tf.variable_scope('seq2seq'):
            # list of [batch_size, frame_feat_dim]
            encoder_inputs = tf.unstack(self.frames, axis=1)
            decoder_inputs = tf.unstack(self.captions, axis=1)
            targets = [x for x in decoder_inputs]
            
            loss_weights = tf.unstack(self.loss_weights, axis=1)

            seq_len = tf.fill([self.batch_size], self.frame_steps)

            with tf.variable_scope('frame_embedding'):
                w_frame_embed = tf.get_variable("w_frame_embed", [self.frame_feat_dim, 2 * self.dim_hidden], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b_frame_embed = tf.get_variable("b_frame_embed", [2 * self.dim_hidden], initializer=tf.constant_initializer(0.0))
                emb_encoder_inputs = [tf.nn.xw_plus_b(x, w_frame_embed, b_frame_embed)
                                     for x in encoder_inputs]

            with tf.variable_scope('embedding'), tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    "embedding", [self.vocab_size, self.dim_hidden], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-4))
                emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                                     for x in decoder_inputs]
            
            for layer_i in xrange(self.enc_layers):
                with tf.variable_scope('encoder{0}'.format(layer_i)):
                    cell_fw = tf.contrib.rnn.LSTMCell(
                        self.dim_hidden,
                        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                        state_is_tuple=False)
                    cell_bw = tf.contrib.rnn.LSTMCell(
                        self.dim_hidden,
                        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                        state_is_tuple=False)
                    (emb_encoder_inputs, fw_state, _) = tf.contrib.rnn.static_bidirectional_rnn(
                        cell_fw, cell_bw, encoder_inputs, dtype=tf.float32,
                        sequence_length=seq_len)
            encoder_outputs = emb_encoder_inputs

            with tf.variable_scope('output_projection'):
                w = tf.get_variable(
                    'w', [self.dim_hidden, self.vocab_size], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4))
                w_t = tf.transpose(w)
                v = tf.get_variable(
                    'v', [self.vocab_size], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4))

            self.scheduled_sampling_prob = tf.minimum(self.global_step / self.scheduled_sampling_decay, 1.0)

            with tf.variable_scope('decoder'):
                # When decoding, use model output from the previous step
                # for the next step.
                loop_function = extract_argmax_and_embed_with_scheduled(
                    embedding, targets, (w, v), update_embedding=True, scheduled_prob=self.scheduled_sampling_prob)

                cell = tf.contrib.rnn.LSTMCell(
                    self.dim_hidden,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                    state_is_tuple=False)

                encoder_outputs = [tf.reshape(x, [self.batch_size, 1, 2 * self.dim_hidden])
                                for x in encoder_outputs]

                self._enc_top_states = tf.concat(axis=1, values=encoder_outputs)
                self._dec_in_state = fw_state
                # During decoding, follow up _dec_in_state are fed from beam_search.
                # dec_out_state are stored by beam_search for next step feeding.
                initial_state_attention = (self.mode == 'decode')
                decoder_outputs, self._dec_out_state = tf.contrib.legacy_seq2seq.attention_decoder(
                    emb_decoder_inputs, self._dec_in_state, self._enc_top_states,
                    cell, num_heads=1, loop_function=loop_function,
                    initial_state_attention=initial_state_attention)

                with tf.variable_scope('output'):
                    model_outputs = []
                    for i in xrange(len(decoder_outputs)):
                        if i > 0:
                            tf.get_variable_scope().reuse_variables()
                        model_outputs.append(
                            tf.nn.xw_plus_b(decoder_outputs[i], w, v))

                if self.mode == 'decode':
                    with tf.variable_scope('decode_output'), tf.device('/cpu:0'):
                        best_outputs = [tf.argmax(x, 1) for x in model_outputs]
                        tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
                        self._outputs = tf.concat(
                            axis=1, values=[tf.reshape(x, [self.batch_size, 1]) for x in best_outputs])

                        self._topk_log_probs, self._topk_ids = tf.nn.top_k(
                            tf.log(tf.nn.softmax(model_outputs[-1])), self.batch_size * 2)


                with tf.variable_scope('loss'):
                    def sampled_loss_func(inputs, labels):
                        # with tf.device('/cpu:0'):  # Try gpu.
                        labels = tf.reshape(labels, [-1, 1])
                        return tf.nn.sampled_softmax_loss(
                            weights=w_t, biases=v, labels=labels, inputs=inputs,
                            num_sampled=self.num_softmax_samples, num_classes=self.vocab_size)

                if self.num_softmax_samples != 0 and self.mode == 'train':
                    self.loss = seq2seq_lib.sampled_sequence_loss(
                        decoder_outputs, targets, loss_weights, sampled_loss_func)
                else:
                    self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                        model_outputs, targets, loss_weights)
                tf.summary.scalar('loss', tf.minimum(12.0, self.loss))

    def add_train_op(self):
        """Sets self._train_op, op to run for training."""

        self.lr_rate = tf.maximum(
            self.min_lr,  # min_lr_rate.
            tf.train.exponential_decay(self.lr, self.global_step, 30000, 0.98))

        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(
            tf.gradients(self.loss, tvars), self.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
        tf.summary.scalar('learning rate', self.lr_rate) 
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step')
        
    def encode_top_state(self, sess, enc_inputs):
        """Return the top states from encoder for decoder.
        Args:
        sess: tensorflow session.
        enc_inputs: encoder inputs of shape [batch_size, enc_timesteps].
        enc_len: encoder input length of shape [batch_size]
        Returns:
        enc_top_states: The top level encoder states.
        dec_in_state: The decoder layer initial state.
        """
        results = sess.run([self._enc_top_states, self._dec_in_state],
                        feed_dict={self.frames: enc_inputs})
        return results[0], results[1][0]

    def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states):
        """Return the topK results and new decoder states."""
        feed = {
            self._enc_top_states: enc_top_states,
            self._dec_in_state:
                np.squeeze(np.array(dec_init_states)),
            self.captions:
                np.transpose(np.array([latest_tokens]))}

        results = sess.run(
            [self._topk_ids, self._topk_log_probs, self._dec_out_state],
            feed_dict=feed)

        ids, probs, states = results[0], results[1], results[2]
        new_states = [s for s in states]
        return ids, probs, new_states

    def build_graph(self):
        self.add_placeholders()
        self.add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.mode == 'train':
            self.add_train_op()
        self.summaries = tf.summary.merge_all()

    def run_train_step(self, sess, frames, captions, loss_weights):
        to_return = [self.train_op, self.summaries, self.loss, self.global_step]
        return sess.run(to_return, feed_dict={self.frames: frames,
                                              self.captions: captions,
                                              self.loss_weights: loss_weights})
