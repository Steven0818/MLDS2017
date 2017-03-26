import tensorflow as tf
import numpy as np

class biGRU_model(object):
    def __init__(self, num_steps=40, vocab_size=30000, num_hidden=800, num_layers=2):
        
        ### input.shape = (batch_size, num_steps+1, vocab_size)
        self.sentences = tf.placeholder(shape=(None, num_steps, vocab_size), dtype=tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        
        final_output = self.sentences[:,1:-1,:]
        
        fwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        fwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(fwGRU_cell, output_keep_prob=self.keep_prob)
        fw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([fwGRU_withDropout] * num_layers)
        
        bwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        bwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(bwGRU_cell, output_keep_prob=self.keep_prob)
        bw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([bwGRU_withDropout] * num_layers)
        
        batch_size = tf.shape(self.sentences)[0]
        sentence_len = tf.fill([batch_size], num_steps)
        
        outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                                cell_fw=fwGRU_cell,
                                cell_bw=bwGRU_cell,
                                dtype=tf.float32,
                                sequence_length = sentence_len,
                                inputs=self.sentences)  
        
        fw_output, bw_output = outputs
        
        ### fw_output.shape = (batch_size, num_steps, num_hidden) and means the 2nd~(num_steps+1)th word
        ### bw_output.shape = (batch_size, num_steps, num_hidden) and means the 0th~(num_steps-1)th word
        ### consider only 2nd~(num_steps-1)th word
        fw_output = fw_output[:,:-2,:]
        bw_output = bw_output[:,2:,:]
        
        ### [batch_size * (num_steps-1), 2*num_hidden]
        gru_output_feature = tf.reshape(tf.concat([fw_output, bw_output], 2), [-1, 2*num_hidden])
        
        with tf.variable_scope('biGRU'):
            
            with tf.device('/cpu:0'):
            ## DNN parameter
                w_1 = tf.get_variable("w_dnn_1", [2*num_hidden, 6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b_1 = tf.get_variable("b_dnn_1", [6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    
                w_2 = tf.get_variable("w_dnn_2", [6*num_hidden, vocab_size], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b_2 = tf.get_variable("b_dnn_2", [vocab_size], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))

     	### [batch_size * (num_steps-1), 6*width]
        dnn_output_1 = tf.nn.relu(tf.matmul(gru_output_feature, w_1) + b_1)

     	### [batch_size * (num_steps-1), vocab_size]
        prediction = tf.matmul(dnn_output_1, w_2) + b_2

     	### [batch_size * (num_steps-1), vocab_size]
        label = tf.reshape(final_output, [-1, vocab_size])

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=label))

     	### [batch_size, num_steps-1, vocab_size]
        self.predict_result = tf.reshape(prediction, [batch_size, num_steps-2, vocab_size] )
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost, global_step=self.global_step)
        config = tf.ConfigProto(log_device_placement = False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def train(self,input_sentences, keep_prob=0.5):
        _,loss = self.sess.run([self.train_op,self.cost],feed_dict={self.sentences:input_sentences, self.keep_prob:keep_prob})
        return loss

    def predict(self,input_sentences, keep_prob=1.):
        return self.sess.run(self.predict_result,feed_dict={self.sentences:input_sentences, self.keep_prob:keep_prob})
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer()) 

    def saveModel(self):
        global_step = self.sess.run(self.global_step)
        saver = tf.train.Saver()
        saver.save(self.sess, './model/model_%d.ckpt' % (global_step))
        
    def loadModel(self, model_path):
        saver = tf.train.Saver(restore_sequentially=True)
        saver.restore(self.sess, model_path)
    
class biGRU_nce_model(object):
    def __init__(self, num_steps=40, vocab_size=30000, num_hidden=800, num_layers=2):
        
        ### input.shape = (batch_size, num_steps+1, vocab_size)
        self.sentences = tf.placeholder(shape=(None, num_steps, vocab_size), dtype=tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.output_widx = tf.arg_max(self.sentences[:,1:-1,:], 2)
        
        fwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        fwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(fwGRU_cell, output_keep_prob=self.keep_prob)
        fw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([fwGRU_withDropout] * num_layers)
        
        bwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        bwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(bwGRU_cell, output_keep_prob=self.keep_prob)
        bw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([bwGRU_withDropout] * num_layers)
        
        batch_size = tf.shape(self.sentences)[0]
        sentence_len = tf.fill([batch_size], num_steps)
        
        outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                                cell_fw=fwGRU_cell,
                                cell_bw=bwGRU_cell,
                                dtype=tf.float32,
                                sequence_length = sentence_len,
                                inputs=self.sentences)  
        
        fw_output, bw_output = outputs
        
        ### fw_output.shape = (batch_size, num_steps, num_hidden) and means the 2nd~(num_steps+1)th word
        ### bw_output.shape = (batch_size, num_steps, num_hidden) and means the 0th~(num_steps-1)th word
        ### consider only 2nd~(num_steps-1)th word
        fw_output = fw_output[:,:-2,:]
        bw_output = bw_output[:,2:,:]
        
        ### [batch_size * (num_steps-1), 2*num_hidden]
        gru_output_feature = tf.reshape(tf.concat([fw_output, bw_output], 2), [-1, 2*num_hidden])
        
        with tf.variable_scope('biGRU'):
            
            with tf.device('/cpu:0'):
            ## DNN parameter
                w_1 = tf.get_variable("w_dnn_1", [2*num_hidden, 6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b_1 = tf.get_variable("b_dnn_1", [6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    
                w_2 = tf.get_variable("w_dnn_2", [vocab_size, 6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b_2 = tf.get_variable("b_dnn_2", [vocab_size], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))

     	### [batch_size * (num_steps-1), 6*width]
        dnn_output_1 = tf.nn.relu(tf.matmul(gru_output_feature, w_1) + b_1)
        prediction = tf.matmul(dnn_output_1, tf.transpose(w_2)) + b_2
        
        out_widx = tf.reshape(self.output_widx, [-1,1])
        
        self.cost = tf.reduce_mean(
              tf.nn.nce_loss(weights=w_2,
                             biases=b_2,
                             labels=out_widx,
                             inputs=dnn_output_1,
                             num_sampled=300,
                             num_classes=vocab_size))
              
     	### [batch_size, num_steps-1, vocab_size]
        self.predict_result = tf.reshape(prediction, [batch_size, num_steps-2, vocab_size] )
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost, global_step=self.global_step)
        config = tf.ConfigProto(log_device_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def train(self, input_sentences, keep_prob=0.5):
        _,loss = self.sess.run([self.train_op,self.cost],feed_dict={self.sentences:input_sentences, self.keep_prob:keep_prob})
        return loss

    def predict(self, input_sentences, keep_prob=1.):
        return self.sess.run(self.predict_result,feed_dict={self.sentences:input_sentences, self.keep_prob:keep_prob})
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer()) 
        
class onewayLSTM_model():
    def __init__(self, num_steps=40, vocab_size=30000, num_hidden=800, num_layers=2):
        
        ### input.shape = (batch_size, num_steps, vocab_size)
        self.sentences = tf.placeholder(shape=(None, num_steps), dtype=tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        
        final_output = self.sentences[:,1:]
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, num_hidden], dtype=tf.float32)
            input = tf.nn.embedding_lookup(embedding, self.sentences)
        
        dropout_input = tf.nn.dropout(input, self.keep_prob)
        
        with tf.variable_scope("LSTM"):
            RNN_cell = tf.contrib.rnn.LSTMCell(num_hidden)
            RNN_withDropout = tf.contrib.rnn.DropoutWrapper(RNN_cell, output_keep_prob=self.keep_prob)
            multiRNN_cell = tf.contrib.rnn.MultiRNNCell([RNN_withDropout] * num_layers)
        
        batch_size = tf.shape(self.sentences)[0]
        
        self._init_state = multiRNN_cell.zero_state(batch_size, tf.float32)
        state = self._init_state
        
        outputs = []
        with tf.variable_scope('LSTM'):
            for time_step in range(num_steps):
                
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()  
                cell_output, state = multiRNN_cell(dropout_input[:, time_step, :], state)
                outputs.append(cell_output)
        
        output = tf.reshape(tf.concat(outputs, 1), [batch_size, num_steps, num_hidden])
        output = output[:,:-1,:]
        output = tf.reshape(output, [-1, num_hidden])
        
        ### [batch_size * (num_steps-1), 2*num_hidden]
        
        
        with tf.variable_scope('LSTM'):
            
            with tf.device('/cpu:0'):
            ## DNN parameter
                w_1 = tf.get_variable("w_dnn_1", [num_hidden, vocab_size], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b_1 = tf.get_variable("b_dnn_1", [vocab_size], initializer= tf.constant_initializer(0.0))
                
        prediction = tf.matmul(output, w_1) + b_1

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    [prediction],
                    [tf.reshape(final_output, [-1])],
                    [tf.ones([batch_size * (num_steps - 1)], dtype=tf.float32)])
        
        self.cost = tf.reduce_mean(loss)
        self._final_state = state
        
     	### [batch_size, num_steps-1, vocab_size]
        self.predict_result = tf.reshape(prediction, [batch_size, num_steps-1, vocab_size] )
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost, global_step=self.global_step)
        config = tf.ConfigProto(log_device_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def train(self,input_sentences, keep_prob=0.5):
        _,loss = self.sess.run([self.train_op,self.cost],feed_dict={self.sentences:input_sentences, self.keep_prob:keep_prob})
        return loss

    def predict(self,input_sentences, keep_prob=1.):
        return self.sess.run(self.predict_result,feed_dict={self.sentences:input_sentences, self.keep_prob:keep_prob})
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())     
        
class embedding_biGRU_model(object):
    def __init__(self, num_steps=40, embedding_dim=300, vocab_size=30000, num_hidden=800, num_layers=2):
        
        ### input.shape = (batch_size, num_steps, embedding_dim)
        self.sentences = tf.placeholder(shape=(None, num_steps, embedding_dim), dtype=tf.float32)
        self.output = tf.placeholder(shape=(None, num_steps), dtype=tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        
        final_output = self.output[:,1:-1]
        
        fwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        fwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(fwGRU_cell, output_keep_prob=self.keep_prob)
        fw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([fwGRU_withDropout] * num_layers)
        
        bwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        bwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(bwGRU_cell, output_keep_prob=self.keep_prob)
        bw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([bwGRU_withDropout] * num_layers)
        
        batch_size = tf.shape(self.sentences)[0]
        sentence_len = tf.fill([batch_size], num_steps)
        
        outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                                cell_fw=fwGRU_cell,
                                cell_bw=bwGRU_cell,
                                dtype=tf.float32,
                                sequence_length = sentence_len,
                                inputs=self.sentences)  
        
        fw_output, bw_output = outputs
        
        ### fw_output.shape = (batch_size, num_steps, num_hidden) and means the 2nd~(num_steps+1)th word
        ### bw_output.shape = (batch_size, num_steps, num_hidden) and means the 0th~(num_steps-1)th word
        ### consider only 2nd~(num_steps-1)th word
        fw_output = fw_output[:,:-2,:]
        bw_output = bw_output[:,2:,:]
        
        ### [batch_size * (num_steps-1), 2*num_hidden]
        gru_output_feature = tf.reshape(tf.concat([fw_output, bw_output], 2), [-1, 2*num_hidden])
        
        with tf.variable_scope('biGRU'):
            
            with tf.device('/cpu:0'):
            ## DNN parameter
                w_1 = tf.get_variable("w_dnn_1", [2*num_hidden, 6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b_1 = tf.get_variable("b_dnn_1", [6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    
                w_2 = tf.get_variable("w_dnn_2", [6*num_hidden, vocab_size], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
                b_2 = tf.get_variable("b_dnn_2", [vocab_size], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))

        dnn_output_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(gru_output_feature, w_1) + b_1), self.keep_prob)
        prediction = tf.matmul(dnn_output_1, w_2) + b_2

     	### [batch_size * (num_steps-2), embedding_dim]
        
        
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [prediction],
            [tf.reshape(final_output, [-1])],
            [tf.ones([batch_size * (num_steps-2)], dtype=tf.float32)])
        
        self.cost = tf.reduce_mean(loss)

     	### [batch_size, num_steps-2, embedding_dim]
        self.predict_result = tf.reshape(prediction, [batch_size, num_steps-2, vocab_size] )
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost, global_step=self.global_step)
        config = tf.ConfigProto(log_device_placement = False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def train(self, input_sentences, outputidx, keep_prob=0.5):
        _,loss = self.sess.run([self.train_op,self.cost],feed_dict={self.sentences:input_sentences, 
                                                                    self.output:outputidx, 
                                                                    self.keep_prob:keep_prob})
        return loss

    def predict(self,input_sentences, keep_prob=1.):
        return self.sess.run(self.predict_result, feed_dict={self.sentences:input_sentences, 
                                                             self.output:np.zeros(input_sentences.shape[:-1]), 
                                                             self.keep_prob:keep_prob})
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer()) 
