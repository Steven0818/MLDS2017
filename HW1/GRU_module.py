import tensorflow as tf

class GRU(object):
	def __init__(self,inputs,num_layers,num_hidden_neurons,output_size,dropout,activation,name):


		inputs_shape = inputs.get_shape()
		#inputs_shape(batch, max_steps, input_feature_dim)

		fGRUcell = tf.nn.rnn_cell.GRUCell(num_hidden_neurons)  # Or LSTMCell(num_neurons)
		fGRUcell_withDropout = tf.nn.rnn_cell.DropoutWrapper(fGRUcell, output_keep_prob=dropout)
		Multi_fGRUcell_withDropout = tf.nn.rnn_cell.MultiRNNCell([fGRUcell_withDropout] * num_layers)

		bGRUcell = tf.nn.rnn_cell.GRUCell(num_hidden_neurons)  # Or LSTMCell(num_neurons)
		bGRUcell_withDropout = tf.nn.rnn_cell.DropoutWrapper(bGRUcell, output_keep_prob=dropout)
		Multi_bGRUcell_withDropout = tf.nn.rnn_cell.MultiRNNCell([reverse_GRUcell_withDropout] * num_layers)



		output_h, state_c = tf.nn.bidirectional_dynamic_rnn(
		                                				cell_fw=Multi_fGRUcell_withDropout,
		                                				cell_bw=Multi_bGRUcell_withDropout,
		                                				dtype=tf.float32,
		                                				inputs=inputs
		                                				) 


		output = tf.concat(2, output_h)
		output_h = tf.reshape(output_h, [-1, num_hidden_neurons])


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