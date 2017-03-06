import tensorflow as tf


def BiGRU(inputs,num_layers,num_hidden_neurons,output_size,dropout,activation,name,sequence_length=None):

	inputs_shape = inputs.get_shape().as_list()
	#inputs_shape(batch, max_steps, input_feature_dim)

	if sequence_length == None:
		sequence_length = [inputs_shape[1]]*inputs_shape[0]
		
	fGRUcell = tf.nn.rnn_cell.GRUCell(num_hidden_neurons)  # Or LSTMCell(num_neurons)
	fGRUcell_withDropout = tf.nn.rnn_cell.DropoutWrapper(fGRUcell, output_keep_prob=dropout)
	Multi_fGRUcell_withDropout = tf.nn.rnn_cell.MultiRNNCell([fGRUcell_withDropout] * num_layers)

	bGRUcell = tf.nn.rnn_cell.GRUCell(num_hidden_neurons)  # Or LSTMCell(num_neurons)
	bGRUcell_withDropout = tf.nn.rnn_cell.DropoutWrapper(bGRUcell, output_keep_prob=dropout)
	Multi_bGRUcell_withDropout = tf.nn.rnn_cell.MultiRNNCell([bGRUcell_withDropout] * num_layers)


	output_h, state_c = tf.nn.bidirectional_dynamic_rnn(
		                                				cell_fw=Multi_fGRUcell_withDropout,
		                                				cell_bw=Multi_bGRUcell_withDropout,
		                                				dtype=tf.float32,
		                                				sequence_length=sequence_length,
		                                				inputs=inputs
		                                				) 


	output = tf.concat(2, output_h)
	output_h = tf.reshape(output_h, [-1, 2*num_hidden_neurons])

	with tf.variable_scope('GRU_'+name):
		weights = tf.get_variable("weights", [2*num_hidden_neurons,output_size],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
		bias = tf.get_variable("bias", [output_size],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
		

	if activation == 'tanh':
		output = tf.tanh(tf.matmul(output_h,weights)+bias)
	elif activation == 'sigmoid':
		output = tf.sigmoid(tf.matmul(output_h,weights)+bias)
	elif activation == 'relu':
		output = tf.nn.relu(tf.matmul(output_h,weights)+bias)
	else:
		output = tf.nn.relu(tf.matmul(output_h,weights)+bias)

	output = tf.reshape(output,[inputs_shape[0],inputs_shape[1],output_size])
	return output


def BiLSTM(inputs,num_layers,num_hidden_neurons,output_size,dropout,activation,name,sequence_length=None,peephole=False):

	inputs_shape = inputs.get_shape().as_list()
	#inputs_shape(batch, max_steps, input_feature_dim)

	if sequence_length == None:
		sequence_length = [inputs_shape[1]]*inputs_shape[0]
		
	fLSTMcell = tf.nn.rnn_cell.LSTMCell(num_hidden_neurons, use_peepholes=peephole)  # Or LSTMCell(num_neurons)
	fLSTMcell_withDropout = tf.nn.rnn_cell.DropoutWrapper(fLSTMcell, output_keep_prob=dropout)
	Multi_fLSTMcell_withDropout = tf.nn.rnn_cell.MultiRNNCell([fLSTMcell_withDropout] * num_layers)

	bLSTMcell = tf.nn.rnn_cell.LSTMCell(num_hidden_neurons, use_peepholes=peephole)  # Or LSTMCell(num_neurons)
	bLSTMcell_withDropout = tf.nn.rnn_cell.DropoutWrapper(bLSTMcell, output_keep_prob=dropout)
	Multi_bLSTMcell_withDropout = tf.nn.rnn_cell.MultiRNNCell([bLSTMcell_withDropout] * num_layers)


	output_h, state_c = tf.nn.bidirectional_dynamic_rnn(
		                                				cell_fw=Multi_fLSTMcell_withDropout,
		                                				cell_bw=Multi_bLSTMcell_withDropout,
		                                				dtype=tf.float32,
		                                				sequence_length=sequence_length,
		                                				inputs=inputs
		                                				) 


	output = tf.concat(2, output_h)
	output_h = tf.reshape(output_h, [-1, 2*num_hidden_neurons])

	with tf.variable_scope('LSTM_'+name):
		weights = tf.get_variable("weights", [2*num_hidden_neurons,output_size],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
		bias = tf.get_variable("bias", [output_size],initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
		

	if activation == 'tanh':
		output = tf.tanh(tf.matmul(output_h,weights)+bias)
	elif activation == 'sigmoid':
		output = tf.sigmoid(tf.matmul(output_h,weights)+bias)
	elif activation == 'relu':
		output = tf.nn.relu(tf.matmul(output_h,weights)+bias)
	else:
		output = tf.nn.relu(tf.matmul(output_h,weights)+bias)

	output = tf.reshape(output,[inputs_shape[0],inputs_shape[1],output_size])
	return output