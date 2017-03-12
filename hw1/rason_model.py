import tensorflow as tf

class biLSTM_model(object):

    def __init__(self, input, batch_size, vocab_size, num_steps, num_hidden=800, dropout_rate=0.5, num_layers=2,):
        
        ### input.shape = (batch_size, num_steps+1, vocab_size)
        self.sentences = tf.placeholder(shape=(None, num_steps + 1, vocab_size), dtype=tf.float32)
        
        fw_input = self.sentence[:,:num_steps, :]
        bw_input = tf.reverse(self.sentence[:,1:,:], 1)
        final_output = self.sentence[:,1:-1,:]
        
        with tf.variable_scope('biGRU'):
            
            fwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
            fwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(fwGRU_cell, output_keep_prob=dropout_rate)
            fw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([fwGRU_withDropout] * num_layers)
                       
            bwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
            bwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(bwGRU_cell, output_keep_prob=dropout_rate)
            bw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([bwGRU_withDropout] * num_layers)
        
        self._fw_initial_state = fw_multiGRU_cell.zero_state(batch_size, tf.float32)
        self._bw_initial_state = bw_multiGRU_cell.zero_state(batch_size, tf.float32)
        
        fw_outputs = []
        bw_outputs = []
        
        fw_state = self._fw_initial_state
        bw_state = self._bw_initial_state

        with tf.variable_scope('biGRU'):
            for time_step in range(num_steps):
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()
                
                fw_cell_output, fw_state = fw_multiGRU_cell(fw_input[:, time_step, :], fw_state)
                fw_outputs.append(fw_cell_output)
                
                bw_cell_output, bw_state = bw_multiGRU_cell(bw_input[:, time_step, :], bw_state)
                bw_outputs.append(bw_cell_output)
        
        
        ### fw_output.shape = (batch_size, num_steps, vocab_size) and means the 2nd~(num_steps+1)th word
        ### bw_output.shape = (batch_size, num_steps, vocab_size) and means the (num_steps)th~1st word
        ### consider only 2nd~(num_steps)th word in fw_output
        ### consider only (num_steps)th~2nd word in bw_output
        fw_output = tf.concat(fw_outputs, 1)[:,:-1,:]
        bw_output = tf.concat(bw_outputs, 1)[:,:-1,:]
        reverse_bw_output = tf.reverse(bw_output, 1)
        
        ### [batch_size * (num_steps-1), 2*num_hidden]
        gru_output_feature = tf.reshape(tf.concat([fw_output, reverse_bw_output], 2), [-1, 2*num_hidden])
        
        with tf.variable_scope('biGRU'):
     	
            ## DNN parameter
            w_1 = tf.get_variable("w_dnn_1", [2*num_hidden, 6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
            b_1 = tf.get_variable("b_dnn_1", [6*num_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))

            w_2 = tf.get_variable("w_dnn_2", [6*num_hidden, vocab_size], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
            b_2 = tf.get_variable("b_dnn_2", [vocab_size], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))

     	### [batch_size * (num_steps-1), 6*width]
        dnn_output_1 = tf.relu(tf.matmul(gru_output_feature, w_1) + b_1)

     	### [batch_size * (num_steps-1), vocab_size]
        prediction = tf.matmul(dnn_output_1, w_2) + b_2

     	### [batch_size * (num_steps-1), vocab_size]
        label = tf.reshape(final_output, [-1, vocab_size])

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=label))

     	### [batch_size, num_steps-1, vocab_size]
        self.predict_result = tf.reshape(prediction, [batch_size, num_steps-1, vocab_size] )
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost, global_step=self.global_step)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))

    def train(self,input_sentences):
        _,loss = self.sess.run([self.train_op,self.cost],feed_dict={self.sentences:input_sentences})
        return loss

    def predict(self,input_sentences):
        return self.sess.run(self.predict_result,feed_dict={self.sentences:input_sentences})
    
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())                