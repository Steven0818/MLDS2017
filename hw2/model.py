import tensorflow as tf
import numpy as np
import random

class S2VT_model():
    
    def __init__(self, frame_steps=80.0, frame_feat_dim=4096, caption_steps=45, vocab_size=3000, dim_hidden=300, schedule_sampling_converge=500):
        

        self.frame_steps = frame_steps
        self.frame_feat_dim = frame_feat_dim
        self.caption_steps = caption_steps
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
        self.schedule_sampling_converge = schedule_sampling_converge

        ## Graph input
        self.frame = tf.placeholder(tf.float32, [None, frame_steps, frame_feat_dim])
        self.caption = tf.placeholder(tf.int32, [None, caption_steps+1])
        self.caption_mask = tf.placeholder(tf.float32, [None, caption_steps+1])
        batch_frame = tf.shape(self.frame)[0]
        batch_caption = tf.shape(self.caption)[0]
        tf.Assert(tf.equal(batch_frame, batch_caption), [batch_frame, batch_caption])
        batch_size = batch_frame
        self.train_state = tf.placeholder(tf.bool)

        
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
            ##input shape of cap_lstm2: [batch_size, 2*dim_hidden]
            with tf.variable_scope('cap_lstm'):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output2, cap_state = cap_lstm(tf.concat([padding, output1], 1), cap_state)
        
        ## Decoding stage
        ## Training util
        def train_cap(prev_endcoder_output, prev_decoder_output, prev_state):
            with tf.device('/cpu:0'):
                word_index = tf.argmax(prev_decoder_output, axis=1)
                word_embed = tf.nn.embedding_lookup(embedding, word_index)
                output, state = cap_lstm(
                    tf.concat([word_embed, prev_endcoder_output], 1), prev_state)
                m_state, c_state = state
                return output, m_state, c_state
        def test_cap(prev_encoder_output, prev_decoder_output, prev_state):
            ##  TODO: beam search
            word_index = tf.argmax(prev_decoder_output, axis=1)
            word_embed = tf.nn.embedding_lookup(embedding, word_index)
            output, state = cap_lstm(
                tf.concat([word_embed, prev_encoder_output], 1), prev_state)
            m_state, c_state = state
            return output, m_state, c_state
        output2 = tf.tile(tf.one_hot([4], vocab_size), [batch_size, 1])
        for i in range(caption_steps):
            
            with tf.variable_scope('att_lstm'):
                tf.get_variable_scope().reuse_variables()
                output1, att_state = att_lstm(padding, att_state)
                        
            with tf.variable_scope('cap_lstm'):
                tf.get_variable_scope().reuse_variables()
                    
                # output2, cap_state = test_cap(output1, output2, cap_state)

                output2, m_state, c_state = tf.cond(self.train_state, lambda: train_cap(output1, self.caption[:i], cap_state), lambda: test_cap(output1, output2, cap_state))
                cap_state = (m_state, c_state)
                cap_lstm_outputs.append(output2)
                
        


        output = tf.reshape(tf.concat(cap_lstm_outputs , 1), [-1, dim_hidden]) 

        ## shape (batch_size*caption_steps, vocab_size)               
        onehot_word_logits = tf.nn.xw_plus_b(output, w_word_onehot, b_word_onehot)
        self.predict_result = tf.reshape(tf.argmax(onehot_word_logits[:,2:], 1)+2, [batch_size, caption_steps])
        
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([onehot_word_logits],
                                                                  [tf.reshape(self.caption[:,1:], [-1])],
                                                                  [tf.reshape(self.caption_mask[:,1:], [-1])])
        
        self.cost = tf.reduce_mean(loss)
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost,global_step=self.global_step)
        
        config = tf.ConfigProto(log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)

    def train(self, input_frame, input_caption,input_caption_mask, keep_prob=0.5):
        _,cost = self.sess.run([self.train_op,self.cost],feed_dict={self.frame:input_frame, 
                                                                    self.caption:input_caption, 
                                                                    self.caption_mask:input_caption_mask,
                                                                    self.train_state:True})
        return cost
   
    def predict(self, input_frame):
        padding = np.zeros([input_frame.shape[0], self.caption_steps + 1])
        words = self.sess.run([self.predict_result], feed_dict={self.frame: input_frame,
                                                                self.caption: padding,
                                                                self.train_state: False})
        return words
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
    
    def schedule_sampling(self):
        prob = self.global_step / self.schedule_sampling_converge
        return random.random() > prob


class S2VT_attention_model():
    
    def __init__(self,frame_steps=20, frame_feat_dim=4096, caption_steps=45, vocab_size=3000, dim_hidden=300, schedule_sampling_converge=500):
        
        self.frame_steps = frame_steps
        self.frame_feat_dim = frame_feat_dim
        self.caption_steps = caption_steps
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
    
        ## Graph input
        self.frame = tf.placeholder(tf.float32, [None, frame_steps, frame_feat_dim])    
        self.caption = tf.placeholder(tf.int32, [None,caption_steps+1])
        self.caption_mask = tf.placeholder(tf.float32, [None, caption_steps+1])
        batch_frame = tf.shape(self.frame)[0]
        batch_caption = tf.shape(self.caption)[0]
        tf.Assert(tf.equal(batch_frame, batch_caption), [batch_frame, batch_caption])
        self.batch_size = batch_frame
        self.train_state = tf.placeholder(tf.bool)
        
        self.global_step = tf.Variable(0, trainable=False)
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
        
        ## attention_position_embedding
        wp = tf.get_variable("w_position_emb_1", [self.dim_hidden,self.dim_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        vp = tf.get_variable("w_position_emb_2", [1,self.dim_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))

        ## attention_align_embedding
        wa = tf.get_variable("w_align_emb",[self.dim_hidden,self.dim_hidden],initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))        

        ## attention_align_embedding
        wc = tf.get_variable("w_attention_emb",[2*self.dim_hidden,self.dim_hidden],initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))        


        ## two lstm param
        with tf.variable_scope("att_lstm"):
            att_lstm = tf.contrib.rnn.LSTMCell(dim_hidden)
        with tf.variable_scope("cap_lstm"):
            cap_lstm = tf.contrib.rnn.LSTMCell(dim_hidden)            
        
        att_state = (tf.zeros([self.batch_size, dim_hidden]),tf.zeros([self.batch_size, dim_hidden]))
        cap_state = (tf.zeros([self.batch_size, dim_hidden]),tf.zeros([self.batch_size, dim_hidden]))
        
        padding = tf.zeros([self.batch_size, dim_hidden])
        
        ##################### Computing Graph ########################
        
        frame_flat = tf.reshape(self.frame, [-1, frame_feat_dim])
        frame_embedding = tf.nn.xw_plus_b( frame_flat, w_frame_embed, b_frame_embed )
        frame_embedding = tf.reshape(frame_embedding, [self.batch_size, frame_steps, dim_hidden])        
        
        enc_lstm_outputs = []
        dec_lstm_outputs = []
        ## Encoding stage
        for i in range(frame_steps):

            with tf.variable_scope('att_lstm'):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output1, att_state = att_lstm(frame_embedding[:,i,:], att_state)
            ##input shape of cap_lstm2: [batch_size, 2*dim_hidden]
            with tf.variable_scope('cap_lstm'):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                #tf.get_variable_scope().reuse_variables()
                output2, cap_state = cap_lstm(tf.concat([padding, output1], 1), cap_state)
            enc_lstm_outputs.append(output2)
        
        ## (batch_size,frame_step,dim_hidden)
        enc_lstm_outputs = tf.reshape(tf.concat(enc_lstm_outputs , 1),[self.batch_size,self.frame_steps,self.dim_hidden])
        
        ## Decoding stage
        ## Training util
        def train_cap(input_lstm,prev_endcoder_output,real_ans,prev_decoder_output,global_step,prev_state):
           word_index = tf.argmax(prev_decoder_output,axis = 1)
            
           with tf.device('/cpu:0'):
                word_embed = tf.nn.embedding_lookup(embedding,word_index)
                output, state = input_lstm(
                    tf.concat([word_embed, prev_endcoder_output], 1), prev_state)
                m_state, c_state = state
                return output, m_state, c_state
        def test_cap(input_lstm,prev_encoder_output, prev_decoder_output, prev_state):
            ##  TODO: beam search
            with tf.device('cpu:0'):
                word_index = tf.argmax(prev_decoder_output, axis=1)
                word_embed = tf.nn.embedding_lookup(embedding, word_index)
                output, state = input_lstm(
                    tf.concat([word_embed, prev_encoder_output], 1), prev_state)
                m_state, c_state = state
                return output, m_state, c_state
        output2 = tf.tile(tf.one_hot([4],vocab_size), [self.batch_size, 1])
        ## Decoding stage
        for i in range(caption_steps):
            
            with tf.variable_scope('att_lstm'):
                tf.get_variable_scope().reuse_variables()
                output1, att_state = att_lstm(padding, att_state)
                        
            with tf.variable_scope('cap_lstm'):
                tf.get_variable_scope().reuse_variables()
                output2, m_state, c_state = tf.cond(self.train_state, lambda: train_cap(cap_lstm,output1, self.caption[:,i],output2,self.global_step,cap_state), lambda: test_cap(cap_lstm,output1, output2, cap_state))
                cap_state = (m_state, c_state)
            ## Attention
            #output2 = self.local_attention(output2,enc_lstm_outputs,wp,vp,wa)
            #concat_output = tf.concat([attention_output,output2] , 1)
            #output2 = tf.tanh(tf.matmul(concat_output,wc))  
            dec_lstm_outputs.append(output2)
        
        output = tf.reshape(tf.concat(dec_lstm_outputs , 1), [-1,dim_hidden])              
        onehot_word_logits = tf.nn.xw_plus_b(output, w_word_onehot, b_word_onehot)
        
        self.predict_result = tf.reshape(tf.argmax(onehot_word_logits[:,2:], 1)+2, [self.batch_size, caption_steps])
        
        onehot_word_logits = tf.unstack(tf.reshape(onehot_word_logits,[self.batch_size,caption_steps,vocab_size]),axis = 1)
        
        caption_ans = tf.unstack(self.caption[:,1:],axis = 1) 
  
        caption_ans_mask = tf.unstack(self.caption_mask[:,1:],axis = 1)     
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(onehot_word_logits,
                                                                  caption_ans,
                                                                  caption_ans_mask)
        
        self.cost = tf.reduce_mean(loss)
        #self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.cost, global_step=self.global_step)
        

        config = tf.ConfigProto(log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)

    def train(self, input_frame, input_caption,input_caption_mask, keep_prob=0.5):
        _,cost = self.sess.run([self.train_op,self.cost],feed_dict={self.frame:input_frame, 
                                                                    self.caption:input_caption, 
                                                                    self.caption_mask:input_caption_mask,
                                                                    self.train_state:True})
        return cost
   
    def predict(self, input_frame):
        padding = np.zeros([input_frame.shape[0], self.caption_steps + 1])
        words = self.sess.run([self.predict_result], feed_dict={self.frame: input_frame,
                                                                self.caption: padding,
                                                                self.train_state: False})
        return words
    def initialize(self):
        self.sess.run(tf.global_variables_initializer()) 


    def local_attention(self,decode_vec,encode_vecs,wp,vp,wa):
        

        ## (batch_size,frame_step)
        score = self.align(decode_vec,encode_vecs,wa)
        ## (dim_hidden,batch_size)
        decode_vec_t = tf.transpose(decode_vec,[1,0])
        ## (1,batch_size)
        pos_feature = tf.matmul(vp,tf.tanh(tf.matmul(wp,decode_vec_t)))
        ## (1,batch_size)
        pt = tf.reshape(self.frame_steps*tf.sigmoid(pos_feature),[self.batch_size])
        local_center = tf.round(pt)

        half_window = 2 #tf.constant(4,shape = [1])
        delta = half_window/2
        
        def index_frame(ele):
            frames,center,pt,score = ele
            s = tf.range(self.frame_steps,dtype = tf.float32)
            score = score*tf.exp(-tf.square(s-pt)/(2*delta*delta))
            right = tf.minimum(center+half_window,self.frame_steps)
            left = tf.maximum(center-half_window,0)
            right = tf.cast(right,tf.int32)
            left = tf.cast(left,tf.int32)
            score = tf.expand_dims(score,0)
            attention_vec = tf.matmul(score[:,left:right],frames[left:right,:])
            attention_vec = tf.reshape(attention_vec,[self.dim_hidden])
            return attention_vec
        ## (batch_size,dim_hidden)
        attention_vec = tf.map_fn(index_frame,[encode_vecs,local_center,pt,score],dtype=tf.float32)
        return attention_vec+decode_vec
                                
        
    

    def align(self,decode_vec,encode_vecs,wa):
        ## (batch_size,dim_hidden,frame_step)
        encode_vecs_t = tf.transpose(encode_vecs,[0,2,1])
        ## (batch_size,1,dim_hidden)*(batch_size,dim_hidden,frame_step)
        score = tf.matmul(tf.expand_dims(tf.matmul(decode_vec,wa),1),encode_vecs_t)
        score = tf.reshape(score,[self.batch_size,self.frame_steps])
        ## (batch_size,frame_step)
        
        return score

class Effective_attention_model():
    
    def __init__(self,frame_steps=20, frame_feat_dim=4096, caption_steps=45, vocab_size=3000, dim_hidden=300, schedule_sampling_converge=500):
        
        self.frame_steps = frame_steps
        self.frame_feat_dim = frame_feat_dim
        self.caption_steps = caption_steps
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
    
        ## Graph input
        self.frame = tf.placeholder(tf.float32, [None, frame_steps, frame_feat_dim])    
        self.caption = tf.placeholder(tf.int32, [None,caption_steps+1])
        self.caption_mask = tf.placeholder(tf.float32, [None, caption_steps+1])
        batch_frame = tf.shape(self.frame)[0]
        batch_caption = tf.shape(self.caption)[0]
        tf.Assert(tf.equal(batch_frame, batch_caption), [batch_frame, batch_caption])
        self.batch_size = batch_frame
        self.train_state = tf.placeholder(tf.bool)
        
        self.global_step = tf.Variable(0, trainable=False)
        ## frame Embedding param 
        with tf.variable_scope("frame_embedding"):
            w_frame_embed = tf.get_variable("w_frame_embed", [frame_feat_dim, 2*dim_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
            b_frame_embed = tf.get_variable("b_frame_embed", [2*dim_hidden], initializer=tf.constant_initializer(0.0))
        
        ## word embedding param
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, dim_hidden], dtype=tf.float32)
        
        ## word embedding to onehot param
        w_word_onehot = tf.get_variable("w_word_onehot", [dim_hidden, vocab_size], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        b_word_onehot = tf.get_variable("b_word_onehot", [vocab_size], initializer=tf.constant_initializer(0.0))
        
        ## attention_position_embedding
        wp = tf.get_variable("w_position_emb_1", [self.dim_hidden,self.dim_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        vp = tf.get_variable("w_position_emb_2", [1,self.dim_hidden], initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))

        ## attention_align_embedding
        wa = tf.get_variable("w_align_emb",[self.dim_hidden,self.dim_hidden],initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))        

        ## attention_align_embedding
        wc = tf.get_variable("w_attention_emb",[2*self.dim_hidden,self.dim_hidden],initializer= tf.contrib.layers.xavier_initializer(dtype=tf.float32))        


        ## two lstm param
        with tf.variable_scope("att_lstm"):
            att_lstm = tf.contrib.rnn.LSTMCell(dim_hidden)
        with tf.variable_scope("cap_lstm"):
            cap_lstm = tf.contrib.rnn.LSTMCell(dim_hidden)            
        
        att_state = (tf.zeros([self.batch_size, dim_hidden]),tf.zeros([self.batch_size, dim_hidden]))
        cap_state = (tf.zeros([self.batch_size, dim_hidden]),tf.zeros([self.batch_size, dim_hidden]))
        
        padding = tf.zeros([self.batch_size, dim_hidden])
        
        ##################### Computing Graph ########################
        
        frame_flat = tf.reshape(self.frame, [-1, frame_feat_dim])
        frame_embedding = tf.nn.xw_plus_b( frame_flat, w_frame_embed, b_frame_embed )
        frame_embedding = tf.reshape(frame_embedding, [self.batch_size, frame_steps, 2*dim_hidden])        
        
        enc_lstm_outputs = []
        dec_lstm_outputs = []
        ## Encoding stage
        for i in range(frame_steps):

            with tf.variable_scope('att_lstm'):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output1, att_state = att_lstm(frame_embedding[:,i,:], att_state)
            ##input shape of cap_lstm2: [batch_size, 2*dim_hidden]
            with tf.variable_scope('cap_lstm'):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output2, cap_state = cap_lstm(output1, cap_state)
            enc_lstm_outputs.append(output2)
        
        ## (batch_size,frame_step,dim_hidden)
        enc_lstm_outputs = tf.reshape(tf.concat(enc_lstm_outputs , 1),[self.batch_size,self.frame_steps,self.dim_hidden])
        
        ## Decoding stage
        ## Training util
        def train_cap(input_lstm,real_ans,prev_decoder_output,prev_attention_output,global_step,prev_state):
            
           with tf.device('/cpu:0'):
                #word_index = tf.argmax(prev_decoder_output,axis = 1)
                word_embed = tf.nn.embedding_lookup(embedding,real_ans)
                output, state = input_lstm(
                    tf.concat([word_embed, prev_attention_output], 1), prev_state)
                m_state, c_state = state
                return output, m_state, c_state
        def test_cap(input_lstm, prev_decoder_output, prev_attention_output,prev_state):
            ##  TODO: beam search
            with tf.device('cpu:0'):
                word_index = tf.argmax(prev_decoder_output, axis=1)
                word_embed = tf.nn.embedding_lookup(embedding, word_index)
                output, state = input_lstm(
                    tf.concat([word_embed,prev_attention_output], 1), prev_state)
                m_state, c_state = state
                return output, m_state, c_state
        output2 = tf.tile(tf.one_hot([4],vocab_size), [self.batch_size, 1])
        attention_output = tf.zeros(shape = [self.batch_size,dim_hidden])
        ## Decoding stage
        for i in range(caption_steps):
            
            with tf.variable_scope('att_lstm'):
                tf.get_variable_scope().reuse_variables()
                output1, m_state, c_state = tf.cond(self.train_state, lambda: train_cap(att_lstm, self.caption[:,i],output2,attention_output,self.global_step,cap_state), lambda: test_cap(att_lstm, output2,attention_output,cap_state))
                cap_state = (m_state, c_state)
                        
            with tf.variable_scope('cap_lstm'):
                tf.get_variable_scope().reuse_variables()
                output2, cap_state = cap_lstm(output1,cap_state)
            ## Attention
            attention_output = self.local_attention(output2,enc_lstm_outputs,wp,vp,wa)
            concat_output = tf.concat([attention_output,output2] , 1)
            attention_output = tf.tanh(tf.matmul(concat_output,wc))  
            dec_lstm_outputs.append(attention_output)
        
        output = tf.reshape(tf.concat(dec_lstm_outputs , 1), [-1,dim_hidden])              
        onehot_word_logits = tf.nn.xw_plus_b(output, w_word_onehot, b_word_onehot)
        
        self.predict_result = tf.reshape(tf.argmax(onehot_word_logits[:,2:], 1)+2, [self.batch_size, caption_steps])
        
        onehot_word_logits = tf.unstack(tf.reshape(onehot_word_logits,[self.batch_size,caption_steps,vocab_size]),axis = 1)
        
        caption_ans = tf.unstack(self.caption[:,1:],axis = 1) 
  
        caption_ans_mask = tf.unstack(self.caption_mask[:,1:],axis = 1)     
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(onehot_word_logits,
                                                                  caption_ans,
                                                                  caption_ans_mask)
        
        self.cost = tf.reduce_mean(loss)
        #self.global_step = tf.Variable(0, trainable=False)
        self.train_op = tf.train.AdamOptimizer(learning_rate = 0.0005).minimize(self.cost, global_step=self.global_step)
        

        config = tf.ConfigProto(log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)

    def train(self, input_frame, input_caption,input_caption_mask, keep_prob=0.5):
        _,cost = self.sess.run([self.train_op,self.cost],feed_dict={self.frame:input_frame, 
                                                                    self.caption:input_caption, 
                                                                    self.caption_mask:input_caption_mask,
                                                                    self.train_state:True})
        return cost
   
    def predict(self, input_frame):
        padding = np.zeros([input_frame.shape[0], self.caption_steps + 1])
        words = self.sess.run([self.predict_result], feed_dict={self.frame: input_frame,
                                                                self.caption: padding,
                                                                self.train_state: False})
        return words
    def initialize(self):
        self.sess.run(tf.global_variables_initializer()) 


    def local_attention(self,decode_vec,encode_vecs,wp,vp,wa):
        

        ## (batch_size,frame_step)
        score = self.align(decode_vec,encode_vecs,wa)
        ## (dim_hidden,batch_size)
        decode_vec_t = tf.transpose(decode_vec,[1,0])
        ## (1,batch_size)
        pos_feature = tf.matmul(vp,tf.tanh(tf.matmul(wp,decode_vec_t)))
        ## (1,batch_size)
        pt = tf.reshape(self.frame_steps*tf.sigmoid(pos_feature),[self.batch_size])
        local_center = tf.round(pt)

        half_window = 2 #tf.constant(4,shape = [1])
        delta = half_window/2
        
        def index_frame(ele):
            frames,center,pt,score = ele
            s = tf.range(self.frame_steps,dtype = tf.float32)
            score = score*tf.exp(-tf.square(s-pt)/(2*delta*delta))
            right = tf.minimum(center+half_window,self.frame_steps)
            left = tf.maximum(center-half_window,0)
            right = tf.cast(right,tf.int32)
            left = tf.cast(left,tf.int32)
            score = tf.expand_dims(score,0)
            attention_vec = tf.matmul(score[:,left:right],frames[left:right,:])
            attention_vec = tf.reshape(attention_vec,[self.dim_hidden])
            return attention_vec
        ## (batch_size,dim_hidden)
        attention_vec = tf.map_fn(index_frame,[encode_vecs,local_center,pt,score],dtype=tf.float32)
        return attention_vec+decode_vec
                                
        
    

    def align(self,decode_vec,encode_vecs,wa):
        ## (batch_size,dim_hidden,frame_step)
        encode_vecs_t = tf.transpose(encode_vecs,[0,2,1])
        ## (batch_size,1,dim_hidden)*(batch_size,dim_hidden,frame_step)
        score = tf.matmul(tf.expand_dims(tf.matmul(decode_vec,wa),1),encode_vecs_t)
        score = tf.reshape(score,[self.batch_size,self.frame_steps])
        ## (batch_size,frame_step)
        
        return score
