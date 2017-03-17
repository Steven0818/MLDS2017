import tensorflow as tf

class biGRU_model(object):
    def __init__(self, num_steps=40, vocab_size=30000, num_hidden=800, dropout_rate=0.5, num_layers=2):
        
        ### input.shape = (batch_size, num_steps+1, vocab_size)
        self.sentences = tf.placeholder(shape=(None, num_steps, vocab_size), dtype=tf.float32)
        
        final_output = self.sentences[:,1:-1,:]
        
        fwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        fwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(fwGRU_cell, output_keep_prob=dropout_rate)
        fw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([fwGRU_withDropout] * num_layers)
        
        bwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        bwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(bwGRU_cell, output_keep_prob=dropout_rate)
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
        config = tf.ConfigProto(log_device_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def train(self,input_sentences):
        _,loss = self.sess.run([self.train_op,self.cost],feed_dict={self.sentences:input_sentences})
        return loss

    def predict(self,input_sentences):
        return self.sess.run(self.predict_result,feed_dict={self.sentences:input_sentences})
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer()) 

        
class biGRU_nce_model(object):
    def __init__(self, num_steps=40, vocab_size=30000, num_hidden=800, dropout_rate=0.5, num_layers=2):
        
        ### input.shape = (batch_size, num_steps+1, vocab_size)
        self.sentences = tf.placeholder(shape=(None, num_steps, vocab_size), dtype=tf.float32)
        self.output_widx = tf.arg_max(self.sentences[:,1:-1,:], 2)
        
        fwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        fwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(fwGRU_cell, output_keep_prob=dropout_rate)
        fw_multiGRU_cell = tf.contrib.rnn.MultiRNNCell([fwGRU_withDropout] * num_layers)
        
        bwGRU_cell = tf.contrib.rnn.GRUCell(num_hidden)
        bwGRU_withDropout = tf.contrib.rnn.DropoutWrapper(bwGRU_cell, output_keep_prob=dropout_rate)
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

    def train(self,input_sentences):
        _,loss = self.sess.run([self.train_op,self.cost],feed_dict={self.sentences:input_sentences})
        return loss

    def predict(self,input_sentences):
        return self.sess.run(self.predict_result,feed_dict={self.sentences:input_sentences})
    
    def initialize(self):
        self.sess.run(tf.global_variables_initializer()) 
        