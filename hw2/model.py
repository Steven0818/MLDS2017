import tensorflow as tf
import numpy as np

class S2VT_model():
    
    def __init__(self, batch_size=20, frame_steps=80, frame_feat_dim=4096, caption_steps=45, vocab_size=3000, dim_hidden=300):
        
        self.batch_size = batch_size
        self.frame_steps = frame_steps
        self.frame_feat_dim = frame_feat_dim
        self.caption_steps = caption_steps
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
        
        ## Graph input
        self.frame = tf.placeholder(tf.float32, [batch_size, frame_steps, frame_feat_dim])    
        self.caption = tf.placeholder(tf.int32, [batch_size, caption_steps+1])
        self.caption_mask = tf.placeholder(tf.float32, [batch_size, caption_steps+1])
        
        ## frame Embedding param 
        with tf.variable_scope("frame_embedding"):
            w_frame_embed = tf.get_variable("w_frame_embed", [frame_feat_dim, dim_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
            b_frame_embed = tf.get_variable("b_frame_embed", [dim_hidden], initializer=tf.constant_initializer(0.0))
        
        ## word embedding param
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, dim_hidden], dtype=tf.float32)
        
        ## word embedding to onehot param
        w_word_onehot = tf.get_variable("w_word_onehot", [dim_hidden, vocab_size], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        b_word_onehot = tf.get_variable("b_word_onehot", [vocab_size], initializer=tf.constant_initializer(0.0))
        
        ## two lstm param
        with tf.variable_scope("att_lstm"):
            att_lstm = tf.contrib.rnn.LSTMCell(dim_hidden)
        with tf.variable_scope("cap_lstm"):
            cap_lstm = tf.contrib.rnn.LSTMCell(dim_hidden)            
        
        att_state = (tf.zeros([batch_size, dim_hidden]),tf.zeros([batch_size, dim_hidden]))
        cap_state = (tf.zeros([batch_size, dim_hidden]),tf.zeros([batch_size, dim_hidden]))
        
        padding = tf.zeros([batch_size, dim_hidden])
        
        ##################### Computing Graph ########################
        
        frame_flat = tf.reshape(self.frame, [-1, frame_feat_dim])
        frame_embedding = tf.nn.xw_plus_b( frame_flat, w_frame_embed, b_frame_embed )
        frame_embedding = tf.reshape(frame_embedding, [batch_size, frame_steps, dim_hidden])        
        
        
        cap_lstm_outputs = []
        
        ## Encoding stage
        for i in range(frame_steps):

            with tf.variable_scope('att_lstm'):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    output1, att_state = att_lstm(frame_embedding[:,i,:], att_state)
                else:
                    output1, att_state = att_lstm(frame_embedding[:,i,:], att_state)
            ##input shape of cap_lstm2: [batch_size, 2*dim_hidden]
            with tf.variable_scope('cap_lstm'):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                    output2, cap_state = cap_lstm(tf.concat([padding, output1], 1), cap_state)
                else:
                    output2, cap_state = cap_lstm(tf.concat([padding, output1], 1), cap_state)
        ## Decoding stage        
        
        for i in range(caption_steps):
            
            with tf.device('/cpu:0'):
                current_word_embed = tf.nn.embedding_lookup(embedding, self.caption[:,i])
            
            with tf.variable_scope('att_lstm'):
                tf.get_variable_scope().reuse_variables()
                output1, att_state = att_lstm(padding, att_state)
                        
            with tf.variable_scope('cap_lstm'):
                tf.get_variable_scope().reuse_variables()
                output2, cap_state = cap_lstm(tf.concat([current_word_embed, output1], 1), cap_state)
                
            cap_lstm_outputs.append(output2)

        output = tf.reshape(tf.concat(cap_lstm_outputs , 1), [-1, dim_hidden])                
        onehot_word_logits = tf.nn.xw_plus_b(output, w_word_onehot, b_word_onehot)
        
        self.predict_result = tf.reshape(onehot_word_logits, [batch_size, caption_steps, vocab_size] )
        
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([onehot_word_logits],
                                                                  [tf.reshape(self.caption[:,1:], [-1])],
                                                                  [tf.reshape(self.caption_mask[:,1:], [-1])])
        
        self.cost = tf.reduce_mean(loss)
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost, global_step=self.global_step)
        
        config = tf.ConfigProto(log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)

    def train(self, input_frame, input_caption,input_caption_mask, keep_prob=0.5):
        _,cost = self.sess.run([self.train_op,self.cost],feed_dict={self.frame:input_frame, 
                                                                    self.caption:input_caption, 
                                                                    self.caption_mask:input_caption_mask})
        return cost
    def initialize(self):
        self.sess.run(tf.global_variables_initializer()) 


        
        
        
            
        
            

        