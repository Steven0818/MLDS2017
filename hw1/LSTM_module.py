import tensorflow as tf


def BiLSTM(inputs,num_layers,num_hidden_neurons,output_size,dropout,activation,name,sequence_length=None,reuse=False,peepole=False):

	inputs_shape = inputs.get_shape().as_list()
	#inputs_shape(batch, max_steps, input_feature_dim)

	if sequence_length == None:
		sequence_length = [inputs_shape[1]]*inputs_shape[0]


	with tf.variable_scope('LSTM_'+name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
			###OUTPUT###
			weights = tf.get_variable("weights", [2*num_hidden_neurons,output_size],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
			bias = tf.get_variable("bias", [output_size],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

			###LSTM###
			_output_h, _state_c = tf.nn.bidirectional_dynamic_rnn(
			                                				cell_fw=reuse['rnn_cell'][0],
			                                				cell_bw=reuse['rnn_cell'][1],
			                                				dtype=tf.float32,
			                                				sequence_length=sequence_length,
			                                				inputs=inputs
			                                				) ##(2,batch,max_steps,num_hiden_neurons), (2,layer,batch,num_hidden_neurons)			
		else:

			###OUTPUT###
			weights = tf.get_variable("weights", [2*num_hidden_neurons,output_size],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
			bias = tf.get_variable("bias", [output_size],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

			###LSTM###
			fLSTMcell = tf.nn.rnn_cell.LSTMCell(num_hidden_neurons,use_peepholes=peepole)  # Or LSTMCell(num_neurons)
			fLSTMcell_withDropout = tf.nn.rnn_cell.DropoutWrapper(fLSTMcell, output_keep_prob=dropout)
			Multi_fLSTMcell_withDropout = tf.nn.rnn_cell.MultiRNNCell([fLSTMcell_withDropout] * num_layers)

			bLSTMcell = tf.nn.rnn_cell.LSTMCell(num_hidden_neurons,use_peepholes=peepole)  # Or LSTMCell(num_neurons)
			bLSTMcell_withDropout = tf.nn.rnn_cell.DropoutWrapper(bLSTMcell, output_keep_prob=dropout)
			Multi_bLSTMcell_withDropout = tf.nn.rnn_cell.MultiRNNCell([bLSTMcell_withDropout] * num_layers)

			_output_h, _state_c = tf.nn.bidirectional_dynamic_rnn(
			                                				cell_fw=Multi_fLSTMcell_withDropout,
			                                				cell_bw=Multi_bLSTMcell_withDropout,
			                                				dtype=tf.float32,
			                                				sequence_length=sequence_length,
			                                				inputs=inputs
			                                				) ##(2,batch,max_steps,num_hiden_neurons), (2,layer,batch,num_hidden_neurons)


	output_h = tf.concat(2, _output_h) ##(batch,max_steps,2*num_hiden_neurons)
	output_h = tf.reshape(output_h, [-1, 2*num_hidden_neurons]) ##(batch*max_steps,2*num_hiden_neurons)
		
		

	if activation == 'tanh':
		output = tf.tanh(tf.matmul(output_h,weights)+bias)
	elif activation == 'sigmoid':
		output = tf.sigmoid(tf.matmul(output_h,weights)+bias)
	elif activation == 'relu':
		output = tf.nn.relu(tf.matmul(output_h,weights)+bias)
	else:
		output = tf.nn.relu(tf.matmul(output_h,weights)+bias)

	output = tf.reshape(output,[inputs_shape[0],inputs_shape[1],output_size]) ##(batch,max_steps,output_size)
	if reuse:
		return {'output':output, 'rnn_cell':[reuse['output'][0], reuse['output'][1]]}
	else:	
		return {'output':output, 'rnn_cell':[Multi_fLSTMcell_withDropout, Multi_bLSTMcell_withDropout]}