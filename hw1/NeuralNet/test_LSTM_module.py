import numpy as np
import tensorflow as tf
import LSTM_module as LSTM



input_shape = [50,250,300]
a=np.random.randn(input_shape[0],input_shape[1],input_shape[2])
x = tf.placeholder(shape=(input_shape[0],input_shape[1],input_shape[2]),dtype=tf.float32)
lstm = LSTM.BiLSTM(	inputs=x,
						num_layers=1,
						num_hidden_neurons=128,
						output_size=300,
						dropout=0.5,
						activation='relu',
						name='test',
						sequence_length=None,
						reuse=False
					) #(batch,max_steps,output_size)
lstm2 = LSTM.BiLSTM(	inputs=lstm['output'],
						num_layers=2,
						num_hidden_neurons=128,
						output_size=300,
						dropout=0.5,
						activation='relu',
						name='test',
						sequence_length=None,
						reuse=lstm
					)
lstm_output_shape_use = lstm2['output'].get_shape()
lstm_output_shape_print1 = tf.shape(lstm2['output'])



sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([lstm_output_shape_print1],feed_dict={x:a}))

###HOW TO GET LAST　OUTPUT###
lstm_output = tf.transpose(lstm['output'], [1, 0, 2]) ##(max_steps,batch,output_size)
last = tf.gather(lstm['output'], int(lstm['output'].get_shape()[0]) - 1) ##(batch,output_size)
print(sess.run(tf.shape(last),feed_dict={x:a}))