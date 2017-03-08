import numpy as np
import tensorflow as tf
import GRU_module as GRU



input_shape = [50,250,300]
a=np.random.randn(input_shape[0],input_shape[1],input_shape[2])
x = tf.placeholder(shape=(input_shape[0],input_shape[1],input_shape[2]),dtype=tf.float32)
gru_output = GRU.BiGRU(	inputs=x,
						num_layers=1,
						num_hidden_neurons=128,
						output_size=200,
						dropout=0.5,
						activation='relu',
						name='test',
						sequence_length=None
					) #(batch,max_steps,output_size)
gru_output_shape_use = gru_output[0].get_shape()
gru_output_shape_print1 = tf.shape(gru_output)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([gru_output_shape_print1],feed_dict={x:a}))

###HOW TO GET LAST　OUTPUT###
gru_output = tf.transpose(gru_output, [1, 0, 2]) ##(max_steps,batch,output_size)
last = tf.gather(gru_output, int(gru_output.get_shape()[0]) - 1) ##(batch,output_size)
print(sess.run(tf.shape(last),feed_dict={x:a}))